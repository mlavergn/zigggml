//! Port of `test_falcon` (tests/test-backend-ops.cpp @cd959bae9): one Falcon layer, whole graph.

const std = @import("std");
const mod = @import("module.zig");

/// One Falcon layer plus output norm and LM head, on Q4_0/Q8_0 weights.
pub const TestFalcon = struct {
    const Self = @This();

    const freq_base: f32 = 10000.0;
    const freq_scale: f32 = 1.0;
    const ext_factor: f32 = 0.0;
    const attn_factor: f32 = 1.0;
    const beta_fast: f32 = 32.0;
    const beta_slow: f32 = 1.0;

    // Constructor parameters, in order, with the C++ defaults.
    n_tokens: i32 = 1,

    // State.
    llm: mod.TestLlm = .{},

    /// Sets the Falcon hparams.
    ///
    /// - `self`: the case.
    pub fn setup(self: *Self) void {
        self.llm = .{ .hp = .{
            .n_vocab = 32000,
            .n_embd = 3200,
            .n_head = 50,
            .n_head_kv = 1,
            .n_rot = 64,
            .n_embd_head = 64,
            .n_ff = 8640,
            .f_norm_eps = 1e-5,
            .f_norm_rms_eps = 0.0,
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

    /// Always `FALCON`.
    ///
    /// - `self`: the case.
    /// - `t`: the output.
    /// - `w`: the destination.
    pub fn opDesc(self: *Self, t: *mod.c.ggml_tensor, w: *std.Io.Writer) !void {
        _ = self;
        _ = t;
        try w.writeAll("FALCON");
    }

    /// The default bound for this case.
    ///
    /// - `self`: the case.
    /// - Returns: the bound.
    pub fn maxNmseErr(self: *Self) f64 {
        _ = self;
        return 2e-3;
    }

    /// Fills the positions and the rest as `TestLlm` does.
    ///
    /// - `self`: the case.
    /// - `ti`: the value source.
    /// - `ctx`: the context.
    pub fn initializeTensors(self: *Self, ti: *mod.TensorInit, ctx: *mod.c.ggml_context) void {
        self.llm.initializeTensors(ti, ctx);
    }

    /// Builds the layer: layer norm, fused-QKV NeoX-RoPE attention over a KV
    /// cache and a parallel GELU FFN, output norm and LM head.
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
        const n_embd: usize = hp.n_embd;
        const n_embd_gqa: usize = hp.nEmbdGqa();

        var inp_l = g.newTensor2d(ctx, mod.c.GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos: the positions.
        const inp_pos = g.newTensor1d(ctx, mod.c.GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask: the mask for 1 head, broadcast to all heads.
        const kq_mask = g.newTensor3d(ctx, mod.c.GGML_TYPE_F16, mod.TestLlm.Hparams.n_kv, hp.n_tokens, 1);

        const k_l = g.newTensor1d(ctx, mod.c.GGML_TYPE_F16, 1638400);
        const v_l = g.newTensor1d(ctx, mod.c.GGML_TYPE_F16, 1638400);

        var cur: *mod.c.ggml_tensor = undefined;
        for (0..mod.TestLlm.Hparams.n_layer) |_| {
            // Norm.
            const attn_norm_w = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, hp.n_embd);
            const attn_norm_b = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, hp.n_embd);
            const attn_norm = llm.buildNorm(ctx, inp_l, attn_norm_w, attn_norm_b, .norm);

            // Self-attention.
            {
                cur = attn_norm;

                const wqkv = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd + 2 * hp.nEmbdGqa());

                cur = mod.c.ggml_mul_mat(ctx, wqkv, cur);

                var q_cur: *mod.c.ggml_tensor = mod.c.ggml_cont(ctx, mod.c.ggml_view_2d(ctx, cur, hp.n_embd, hp.n_tokens, cur.nb[1], 0 * @sizeOf(f32) * n_embd));
                var k_cur: *mod.c.ggml_tensor = mod.c.ggml_cont(ctx, mod.c.ggml_view_2d(ctx, cur, hp.nEmbdGqa(), hp.n_tokens, cur.nb[1], 1 * @sizeOf(f32) * n_embd));
                const v_cur: *mod.c.ggml_tensor = mod.c.ggml_cont(ctx, mod.c.ggml_view_2d(ctx, cur, hp.nEmbdGqa(), hp.n_tokens, cur.nb[1], 1 * @sizeOf(f32) * (n_embd + n_embd_gqa)));

                q_cur = mod.c.ggml_reshape_3d(ctx, q_cur, hp.n_embd_head, hp.n_head, hp.n_tokens);
                k_cur = mod.c.ggml_reshape_3d(ctx, k_cur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens);

                // Mode 2: NeoX.
                q_cur = mod.c.ggml_rope_ext(ctx, q_cur, inp_pos, null, n_rot, 2, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                k_cur = mod.c.ggml_rope_ext(ctx, k_cur, inp_pos, null, n_rot, 2, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

                llm.buildKvStore(ctx, k_l, v_l, k_cur, v_cur);

                cur = llm.buildKqv(g, k_l, v_l, q_cur, kq_mask, 1.0 / @sqrt(@as(f32, @floatFromInt(hp.n_embd_head))));
            }

            const ffn_inp = cur;

            // Feed forward.
            {
                const ffn_up = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
                const ffn_down = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q4_0, hp.n_ff, hp.n_embd);
                cur = attn_norm;
                cur = mod.c.ggml_mul_mat(ctx, ffn_up, cur);
                cur = mod.c.ggml_gelu(ctx, cur);
                cur = mod.c.ggml_mul_mat(ctx, ffn_down, cur);
            }

            cur = mod.c.ggml_add(ctx, cur, ffn_inp);
            cur = mod.c.ggml_add(ctx, cur, inp_l);

            // Input for the next layer.
            inp_l = cur;
        }

        cur = inp_l;

        const output_norm = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, hp.n_embd);
        const output_norm_b = g.newTensor1d(ctx, mod.c.GGML_TYPE_F32, hp.n_embd);
        cur = llm.buildNorm(ctx, cur, output_norm, output_norm_b, .norm);

        // LM head.
        const output = g.newTensor2d(ctx, mod.c.GGML_TYPE_Q8_0, hp.n_embd, hp.n_vocab);
        cur = mod.c.ggml_mul_mat(ctx, output, cur);
        return cur;
    }
};
