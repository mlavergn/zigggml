//! Port of `test_llama` (tests/test-backend-ops.cpp @cd959bae9): one Llama layer, whole graph.

const std = @import("std");
const mod = @import("module.zig");

/// One Llama layer plus output norm and LM head, on Q4_0 weights.
pub const TestLlama = struct {
    const Self = @This();

    const freq_base: f32 = 10000.0;
    const freq_scale: f32 = 1.0;
    const ext_factor: f32 = 0.0;
    const attn_factor: f32 = 1.0;
    const beta_fast: f32 = 32.0;
    const beta_slow: f32 = 1.0;

    // Constructor parameters, in order, with the C++ defaults.
    n_tokens: i32 = 1,
    /// Compare the whole graph at once.
    fused: bool = false,

    // State.
    llm: mod.TestLlm = .{},

    /// Sets the Llama hparams.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        self.llm = .{ .hp = .{
            .n_vocab = 32000,
            .n_embd = 3200,
            .n_head = 32,
            .n_head_kv = 32,
            .n_rot = 100,
            .n_embd_head = 100,
            .n_ff = 8640,
            .f_norm_eps = 0.0,
            .f_norm_rms_eps = 1e-5,
            .n_tokens = self.n_tokens,
        } };
    }

    /// Writes the case's parameters.
    ///
    /// - `self`: the case.
    /// - `w`: the destination.
    pub fn vars(self: *const Self, w: *std.Io.Writer) !void {
        var v = mod.Vars.init(w);
        try v.add("n_tokens", self.llm.hp.n_tokens);
    }

    /// Always `LLAMA`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("LLAMA");
    }

    /// The default bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 2e-3;
    }

    /// Whether the graph is compared as a whole.
    ///
    /// - `self`: the case.
    /// - Returns: `fused`.
    pub fn runWholeGraph(self: *Self) bool {
        return self.fused;
    }

    /// Fills the positions and the rest as `TestLlm` does.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        self.llm.initializeTensors(ti, ctx);
    }

    /// Builds the layer: RMS norm, RoPE-ed attention over a KV cache, SiLU
    /// gated FFN, output norm and LM head.
    ///
    /// - `self`: the case.
    /// - `g`: the context wrapper.
    /// - Returns: the logits.
    pub fn buildGraph(self: *Self, g: *mod.Graph) *mod.c.ggml_tensor {
        const ctx = g.ctx;
        const llm = &self.llm;
        const hp = llm.hp;
        const n_rot: c_int = @intCast(hp.n_rot);
        const n_ctx_orig: c_int = mod.TestLlm.Hparams.n_ctx_orig;

        var inp_l = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos: the positions.
        const inp_pos = g.newTensor1d(ctx, mod.c.GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask: the mask for 1 head, broadcast to all heads.
        const kq_mask = g.newTensor3d(ctx, mod.c.GGML_TYPE_F16, mod.TestLlm.Hparams.n_kv, hp.n_tokens, 1);

        const k_l = g.newTensor1d(ctx, mod.c.GGML_TYPE_F16, 1638400);
        const v_l = g.newTensor1d(ctx, mod.c.GGML_TYPE_F16, 1638400);

        var cur: *mod.c.ggml_tensor = undefined;
        for (0..mod.TestLlm.Hparams.n_layer) |_| {
            const inp_sa = inp_l;

            // Norm.
            const attn_norm = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, hp.n_embd);
            cur = llm.buildNorm(ctx, inp_l, attn_norm, null, .normRms);

            // Self-attention.
            {
                const wq = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
                const wk = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q4_0, hp.n_embd, hp.nEmbdGqa());
                const wv = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q4_0, hp.n_embd, hp.nEmbdGqa());

                // Compute Q and K and RoPE them.
                var q_cur: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, wq, cur);
                var k_cur: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, wk, cur);
                const v_cur: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, wv, cur);

                q_cur = mod.c.ggml_rope_ext(ctx, mod.c.ggml_reshape_3d(ctx, q_cur, hp.n_embd_head, hp.n_head, hp.n_tokens), inp_pos, null, n_rot, 0, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                k_cur = mod.c.ggml_rope_ext(ctx, mod.c.ggml_reshape_3d(ctx, k_cur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens), inp_pos, null, n_rot, 0, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

                llm.buildKvStore(ctx, k_l, v_l, k_cur, v_cur);

                cur = llm.buildKqv(g, k_l, v_l, q_cur, kq_mask, 1.0 / @sqrt(@as(f32, @floatFromInt(hp.n_embd_head))));
            }

            const ffn_inp: *mod.c.ggml_tensor = mod.c.ggml_add(ctx, cur, inp_sa);

            // Feed-forward network.
            const ffn_norm = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, hp.n_embd);
            cur = llm.buildNorm(ctx, ffn_inp, ffn_norm, null, .normRms);

            const ffn_gate = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            const ffn_down = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q4_0, hp.n_ff, hp.n_embd);
            const ffn_up = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            const tmp: *mod.c.ggml_tensor = mod.c.ggml_mul_mat(ctx, ffn_up, cur);
            cur = mod.c.ggml_mul_mat(ctx, ffn_gate, cur);
            cur = mod.c.ggml_silu(ctx, cur);
            cur = mod.c.ggml_mul(ctx, cur, tmp);
            cur = mod.c.ggml_mul_mat(ctx, ffn_down, cur);

            cur = mod.c.ggml_add(ctx, cur, ffn_inp);

            // Input for the next layer.
            inp_l = cur;
        }

        cur = inp_l;

        const output_norm = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, hp.n_embd);
        cur = llm.buildNorm(ctx, cur, output_norm, null, .normRms);

        // LM head.
        const output = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q4_0, hp.n_embd, hp.n_vocab);
        cur = mod.c.ggml_mul_mat(ctx, output, cur);
        return cur;
    }
};
