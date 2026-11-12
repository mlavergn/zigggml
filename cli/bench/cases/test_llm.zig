//! Port of `test_llm` (tests/test-backend-ops.cpp @cd959bae9): the shared base of the whole-model cases.

const mod = @import("module.zig");

/// The normalization a block uses (`llm_norm_type`).
pub const LlmNormType = enum {
    /// Layer norm (`LLM_NORM`).
    norm,
    /// RMS norm (`LLM_NORM_RMS`).
    normRms,
};

/// The model and batch shape of a whole-model case (`llama_hparams`).
pub const LlamaHparams = struct {
    const Self = @This();

    /// The number of layers.
    pub const n_layer: u32 = 1;
    /// The user-specified context size (cparams).
    pub const n_ctx: u32 = 512;
    /// The training context size.
    pub const n_ctx_orig: u32 = n_ctx;
    /// The size of the KV cache considered (`n_kv <= n_ctx`).
    pub const n_kv: i32 = 32;
    /// Where new KV data is stored in the cache.
    pub const kv_head: i32 = 1;

    n_vocab: u32 = 0,
    n_embd: u32 = 0,
    n_head: u32 = 0,
    n_head_kv: u32 = 0,
    n_rot: u32 = 0,
    /// The dimension of values (d_v).
    n_embd_head: u32 = 0,
    n_ff: u32 = 0,
    f_norm_eps: f32 = 0.0,
    f_norm_rms_eps: f32 = 0.0,
    /// The batch size.
    n_tokens: i32 = 0,

    /// The dimension of key embeddings across all KV heads.
    ///
    /// - `self`: the hparams.
    /// - Returns: `n_embd_head * n_head_kv`.
    pub fn nEmbdGqa(self: *const Self) u32 {
        return self.n_embd_head * self.n_head_kv;
    }
};

/// The shared state and builders of `TestLlama` and `TestFalcon`, which
/// compose it (upstream's abstract `test_llm`). Not a case by itself.
pub const TestLlm = struct {
    const Self = @This();

    /// The hparams type.
    pub const Hparams = LlamaHparams;
    /// The normalization type.
    pub const NormType = LlmNormType;

    /// The model and batch shape.
    hp: LlamaHparams = .{},

    /// Normalizes `cur`, scales by `mw` and adds `mb` when given
    /// (`llm_build_norm`).
    ///
    /// - `self`: the model.
    /// - `ctx`: the context.
    /// - `cur`: the input.
    /// - `mw`: the scale.
    /// - `mb`: the bias, or null.
    /// - `norm_type`: layer or RMS norm.
    /// - Returns: the output.
    pub fn buildNorm(self: *const Self, ctx: *mod.c.ggml_context, cur: *mod.c.ggml_tensor, mw: *mod.c.ggml_tensor, mb: ?*mod.c.ggml_tensor, norm_type: LlmNormType) *mod.c.ggml_tensor {
        var out: *mod.c.ggml_tensor = switch (norm_type) {
            .norm => mod.c.ggml_norm(ctx, cur, self.hp.f_norm_eps),
            .normRms => mod.c.ggml_rms_norm(ctx, cur, self.hp.f_norm_rms_eps),
        };
        out = mod.c.ggml_mul(ctx, out, mw);
        if (mb) |b| out = mod.c.ggml_add(ctx, out, b);
        return out;
    }

    /// Copies the new K (RoPE-ed) and transposed V into views of the caches
    /// (`llm_build_kv_store`). The copies are not added to the graph, as
    /// upstream's are not.
    ///
    /// - `self`: the model.
    /// - `ctx`: the context.
    /// - `k_l`: the K cache.
    /// - `v_l`: the V cache.
    /// - `k_cur`: the new K.
    /// - `v_cur`: the new V.
    pub fn buildKvStore(self: *const Self, ctx: *mod.c.ggml_context, k_l: *mod.c.ggml_tensor, v_l: *mod.c.ggml_tensor, k_cur: *mod.c.ggml_tensor, v_cur: *mod.c.ggml_tensor) void {
        const n_tokens: i64 = self.hp.n_tokens;
        const n_embd_gqa: i64 = self.hp.nEmbdGqa();
        const kv_head: usize = LlamaHparams.kv_head;

        // The transposed [n_tokens, n_embd] V matrix.
        const v_cur_t: *mod.c.ggml_tensor = mod.c.ggml_transpose(ctx, mod.c.ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens));

        const k_cache_view: *mod.c.ggml_tensor = mod.c.ggml_view_1d(ctx, k_l, n_tokens * n_embd_gqa, mod.c.ggml_row_size(k_l.type, n_embd_gqa) * kv_head);

        const v_es = mod.c.ggml_element_size(v_l);
        const v_cache_view: *mod.c.ggml_tensor = mod.c.ggml_view_2d(ctx, v_l, n_tokens, n_embd_gqa, LlamaHparams.n_ctx * v_es, kv_head * v_es);

        // Stores the RoPE-ed version of K in the KV cache.
        _ = mod.c.ggml_cpy(ctx, k_cur, k_cache_view);
        _ = mod.c.ggml_cpy(ctx, v_cur_t, v_cache_view);
    }

    /// Attention of `q_cur` over the cached K and V, merged across heads and
    /// projected by a new `wo` (`llm_build_kqv`).
    ///
    /// - `self`: the model.
    /// - `g`: the context wrapper; `wo` is created in `g.ctx`.
    /// - `k_l`: the K cache.
    /// - `v_l`: the V cache.
    /// - `q_cur`: the queries.
    /// - `kq_mask`: the attention mask.
    /// - `kq_scale`: the KQ scale.
    /// - Returns: the output.
    pub fn buildKqv(self: *const Self, g: *mod.Graph, k_l: *mod.c.ggml_tensor, v_l: *mod.c.ggml_tensor, q_cur: *mod.c.ggml_tensor, kq_mask: *mod.c.ggml_tensor, kq_scale: f32) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const hp = self.hp;
        const n_ctx: usize = LlamaHparams.n_ctx;
        const q: *mod.c.ggml_tensor = mod.c.ggml_permute(ctx, q_cur, 0, 2, 1, 3);

        const k: *mod.c.ggml_tensor = mod.c.ggml_view_3d(ctx, k_l, hp.n_embd_head, LlamaHparams.n_kv, hp.n_head_kv, mod.c.ggml_row_size(k_l.type, hp.nEmbdGqa()), mod.c.ggml_row_size(k_l.type, hp.n_embd_head), 0);

        var kq: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, k, q);
        kq = mod.c.ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0);

        // Split the cached V into n_head heads.
        const v_es = mod.c.ggml_element_size(v_l);
        const v: *mod.c.ggml_tensor = mod.c.ggml_view_3d(ctx, v_l, LlamaHparams.n_kv, hp.n_embd_head, hp.n_head_kv, v_es * n_ctx, v_es * n_ctx * hp.n_embd_head, 0);

        const kqv: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, v, kq);
        const kqv_merged: *mod.c.ggml_tensor = mod.c.ggml_permute(ctx, kqv, 0, 2, 1, 3);
        var cur: *mod.c.ggml_tensor = mod.c.ggml_cont_2d(ctx, kqv_merged, hp.n_embd_head * hp.n_head, hp.n_tokens);

        const wo = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
        cur = mod.c.ggml_mul_mat(ctx, wo, cur);
        return cur;
    }

    /// Fills the I32 tensors (the positions) with random positions in the
    /// context and the rest uniformly.
    ///
    /// - `self`: the model.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *const Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        const n_tokens: usize = @intCast(self.hp.n_tokens);
        var it = mod.Helpers.tensors(ctx);
        while (it.next()) |t| {
            if (t.type == mod.c.GGML_TYPE_I32) {
                const data = ti.allocator.alloc(i32, n_tokens) catch @panic("OOM");
                defer ti.allocator.free(data);
                for (data) |*d| d.* = @intCast(ti.random.uintLessThan(u32, LlamaHparams.n_ctx));
                mod.c.ggml_backend_tensor_set(t, data.ptr, 0, n_tokens * @sizeOf(i32));
            } else {
                ti.uniform(t, -1.0, 1.0);
            }
        }
    }
};
