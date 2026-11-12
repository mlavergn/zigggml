const std = @import("std");
const log = std.log.scoped(.zigggml_nvml);
const mod = @import("module.zig");

/// The NVML C API, present only in CUDA builds (`build.zig` translates
/// `nvml.h` and links `libnvidia-ml` into the CLI alone).
const nvml_c = if (mod.config.use_cuda) @import("nvml_c") else struct {};

/// GPU utilization over a window, from NVML's utilization samples.
///
/// Only CUDA devices are measured; everything else reports null, printed
/// as `n/a`. Without a CUDA build this compiles to a stub.
pub const Nvml = struct {
    const Self = @This();

    /// Whether `nvmlInit_v2` succeeded.
    ready: bool = false,

    /// Initializes NVML when the build has it.
    ///
    /// - Returns: a handle; inert when NVML is absent or fails to start.
    pub fn init() Self {
        if (!mod.config.use_cuda) return Self{};
        const ok = nvml_c.nvmlInit_v2() == nvml_c.NVML_SUCCESS;
        if (!ok) log.warn("nvmlInit failed; GPU usage will read n/a", .{});
        return Self{ .ready = ok };
    }

    /// Shuts NVML down.
    ///
    /// - `self`: the handle.
    pub fn deinit(self: *Self) void {
        if (!mod.config.use_cuda) return;
        if (self.ready) _ = nvml_c.nvmlShutdown();
        self.ready = false;
    }

    /// A window over which to measure a device.
    pub const Window = struct {
        /// NVML's device handle, as an opaque pointer.
        device: ?*anyopaque = null,
        /// NVML's sample clock (µs since the epoch) at the window's start.
        since: u64 = 0,
    };

    /// Starts a window for the device behind `backend`.
    ///
    /// - `self`: the handle.
    /// - `backend`: the backend under test.
    /// - Returns: the window; its device is null when the backend is not a
    ///   CUDA device NVML knows.
    pub fn begin(self: *Self, backend: mod.c.ggml_backend_t) Window {
        if (!mod.config.use_cuda or !self.ready) return .{};
        const dev = mod.c.ggml_backend_get_device(backend);
        const reg_name = std.mem.span(mod.c.ggml_backend_reg_name(mod.c.ggml_backend_dev_backend_reg(dev)));
        if (!std.mem.eql(u8, reg_name, "CUDA")) return .{};
        var props: mod.c.struct_ggml_backend_dev_props = undefined;
        mod.c.ggml_backend_dev_get_props(dev, &props);
        const bus_id = props.device_id orelse return .{};
        var handle: nvml_c.nvmlDevice_t = null;
        if (nvml_c.nvmlDeviceGetHandleByPciBusId_v2(bus_id, &handle) != nvml_c.NVML_SUCCESS) return .{};
        return .{ .device = handle, .since = Self.latestSample(handle) };
    }

    /// The mean GPU utilization since the window started.
    ///
    /// Averages the samples NVML took after `window.since`; when there are
    /// none (a window shorter than NVML's sampling period), falls back to the
    /// device's current utilization rate.
    ///
    /// - `self`: the handle.
    /// - `window`: from `begin`.
    /// - Returns: the percentage, or null when unmeasurable.
    pub fn end(self: *Self, window: Window) ?f64 {
        if (!mod.config.use_cuda or !self.ready) return null;
        const device: nvml_c.nvmlDevice_t = @ptrCast(window.device orelse return null);
        var samples: [128]nvml_c.nvmlSample_t = undefined;
        var count: c_uint = samples.len;
        var value_type: nvml_c.nvmlValueType_t = undefined;
        const rc = nvml_c.nvmlDeviceGetSamples(device, nvml_c.NVML_GPU_UTILIZATION_SAMPLES, window.since, &value_type, &count, &samples);
        if (rc == nvml_c.NVML_SUCCESS and count > 0) {
            var sum: f64 = 0;
            for (samples[0..count]) |s| sum += @floatFromInt(s.sampleValue.uiVal);
            return sum / @as(f64, @floatFromInt(count));
        }
        var rates: nvml_c.nvmlUtilization_t = undefined;
        if (nvml_c.nvmlDeviceGetUtilizationRates(device, &rates) != nvml_c.NVML_SUCCESS) return null;
        return @floatFromInt(rates.gpu);
    }

    /// The timestamp of the newest utilization sample, so a window counts
    /// only samples taken after it opened.
    fn latestSample(device: nvml_c.nvmlDevice_t) u64 {
        var samples: [128]nvml_c.nvmlSample_t = undefined;
        var count: c_uint = samples.len;
        var value_type: nvml_c.nvmlValueType_t = undefined;
        if (nvml_c.nvmlDeviceGetSamples(device, nvml_c.NVML_GPU_UTILIZATION_SAMPLES, 0, &value_type, &count, &samples) != nvml_c.NVML_SUCCESS) return 0;
        var latest: u64 = 0;
        for (samples[0..count]) |s| latest = @max(latest, s.timeStamp);
        return latest;
    }
};
