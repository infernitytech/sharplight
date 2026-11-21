namespace Sharplight.Runtime;

/// <summary>
/// Provider-neutral names of optional device features. A provider maps its own mechanisms onto these
/// (e.g. Vulkan VK_KHR_cooperative_matrix and CUDA tensor cores / mma.sync provide <see cref="MatrixF16"/>, Metal
/// simdgroup_matrix provides <see cref="MatrixF16Small"/>). Hosts can turn any of them off for a device or a single compilation.
/// </summary>
/// <remarks>
/// Diagnostic environment switches of Sharplight.Core:
/// <list type="bullet">
/// <item><c>SHARPLIGHT_DISABLE_FEATURES</c>: comma-separated feature names (or "all") disabled for every device, see <see cref="DisabledByEnvironment"/>.</item>
/// <item><c>SHARPLIGHT_LIMIT_DEVICE_MEMORY</c>, <c>SHARPLIGHT_LIMIT_HOST_VISIBLE_DEVICE_MEMORY</c>, <c>SHARPLIGHT_LIMIT_MAX_ALLOCATION</c>, <c>SHARPLIGHT_LIMIT_PINNED_HOST_MEMORY</c>: process-wide memory caps, see <see cref="DeviceMemoryLimits"/>.</item>
/// <item><c>SHARPLIGHT_SUBMISSION_BUDGET_MS</c>: process-wide override of <see cref="DeviceInfo.SubmissionBudgetMs"/>; read by the providers.</item>
/// </list>
/// </remarks>
public static class DeviceFeatures
{
    /// <summary>Subgroup/warp-level matrix multiply-accumulate on f16 inputs with f32 accumulation (tensor cores).</summary>
    public const string MatrixF16 = "matrix_f16";
    /// <summary>
    /// Subgroup-level matrix multiply-accumulate on 8x8 f16 fragments with f32 accumulation (Metal: simdgroup_matrix, Apple GPUs
    /// from the A14 / M1). The 16x16 fragments the <see cref="MatrixF16"/> kernels are written for are not available, so only the
    /// kernels written for 8x8 fragments use it (the prefill GEMM); the others take their SIMT fallbacks.
    /// </summary>
    public const string MatrixF16Small = "matrix_f16_8x8";
    /// <summary>Matrix multiply-accumulate on bf16 inputs with f32 accumulation.</summary>
    public const string MatrixBF16 = "matrix_bf16";
    /// <summary>Matrix multiply-accumulate on int8 inputs with int32 accumulation.</summary>
    public const string MatrixInt8 = "matrix_int8";
    /// <summary>
    /// Matrix multiply-accumulate on 8-bit floats (OCP E4M3) with f32 accumulation (Vulkan: VK_EXT_shader_float8 cooperative
    /// matrices with an f8E4m3 x f8E4m3 + f32 shape).
    /// </summary>
    public const string MatrixFp8 = "matrix_fp8";
    /// <summary>
    /// Workgroup-wide f16 matrix multiply-accumulate on large tiles whose operands are loaded straight from memory
    /// by tensor addressing (bounds-clamped slices, transposed views) and decoded on load from block formats
    /// (Vulkan: VK_NV_cooperative_matrix2). Lets the device stage tiles itself instead of the kernel.
    /// </summary>
    public const string MatrixWorkgroup = "matrix_workgroup";
    /// <summary>
    /// The 16x16 f16 subgroup fragments of <see cref="MatrixF16"/> (32-lane subgroups) hold their elements in the layout of
    /// the 16x8x16 MMA instructions: lane l has rows (A) / columns (B) l / 4 and l / 4 + 8 at K positions 2 (l % 4) + {0, 1, 8, 9}.
    /// Kernels can then fill fragments from registers instead of staging them in shared memory. Detected for NVIDIA GPUs;
    /// the kernels that rely on it verify the layout at run time.
    /// </summary>
    public const string MatrixMmaLayout = "matrix_mma_layout";
    /// <summary>Native 16-bit float arithmetic in kernels.</summary>
    public const string Float16 = "fp16";
    /// <summary>Native bfloat16 types in kernels.</summary>
    public const string BFloat16 = "bf16";
    /// <summary>Native 8-bit float types (E4M3 / E5M2) in kernels.</summary>
    public const string Float8 = "fp8";
    /// <summary>Packed int8 dot-product instructions (dp4a-style).</summary>
    public const string IntDot = "int_dot";
    /// <summary>Subgroup/warp collective operations (reductions, shuffles).</summary>
    public const string SubgroupOps = "subgroup_ops";
    /// <summary>The kernel can pin the subgroup size.</summary>
    public const string SubgroupSizeControl = "subgroup_size_control";
    /// <summary>
    /// Large matrix products of 16-bit float weights (f16, and bf16 stored as f16) run on the vendor's GEMM library where it is faster
    /// than the generated kernels, with the generated tensor-core GEMMs' f16 activations and precision (CUDA: cuBLASLt from the optional
    /// <c>Sharplight.Cuda.Blas</c> package). Reported only when the library is installed; disabling it keeps every product on the
    /// generated kernels.
    /// </summary>
    public const string LibraryGemm = "library_gemm";

    /// <summary>
    /// Large matrix products of block-scaled 8-bit float weights (E4M3 in 128-value scale groups) run on a hand-written GEMM of the
    /// provider instead of the generated tensor-core GEMM (CUDA, compute capability 8.9 and later: a CUDA C++ kernel with
    /// asynchronous copies into a multi-stage ring of dynamic shared memory). Disabling it keeps the generated kernel.
    /// </summary>
    public const string NativeGemmFp8 = "native_gemm_fp8";

    /// <summary>
    /// The device copies tiles of a tensor from device memory into a workgroup's shared memory with one asynchronous instruction
    /// described by a tensor map (CUDA: <c>cp.async.bulk.tensor</c>, compute capability 9.0 and later), completing on shared-memory
    /// barriers. The provider's hand-written GEMMs use it for warp-specialized main loops (a copy warp and computing warps);
    /// disabling it keeps their per-thread asynchronous copies.
    /// </summary>
    public const string TensorCopies = "tensor_copies";

    /// <summary>
    /// Large matrix products of 16-bit float weights (f16, and bf16 stored as f16) run on a hand-written f16 tensor-core GEMM of the
    /// provider, with the operands and precision of <see cref="LibraryGemm"/> but without the vendor library (CUDA, compute capability
    /// 8.0 and later); preferred over the library where both exist. Disabling it leaves those products to the library or the
    /// generated GEMMs.
    /// </summary>
    public const string NativeGemmF16 = "native_gemm_f16";

    /// <summary>
    /// The expert-grouped products of a mixture-of-experts layer with MXFP4 expert weights run, for batches with many rows per expert,
    /// on a hand-written grouped GEMM of the provider (asynchronous copies of the raw 4-bit codes into a multi-stage ring of dynamic
    /// shared memory, decoded per tensor-core MMA; CUDA, compute capability 8.0 and later). Disabling it keeps the generated grouped
    /// GEMMs.
    /// </summary>
    public const string NativeGemmMoe = "native_gemm_moe";

    /// <summary>
    /// Causal attention of many queries against an f16 KV cache (prefill, prompt chunks of a serving step) runs on a hand-written
    /// flash-attention kernel of the provider (query fragments, scores, softmax statistics and outputs in registers, key / value
    /// tiles copied asynchronously into a two-stage ring of shared memory; CUDA, compute capability 8.0 and later), including
    /// sliding windows, chunked attention, soft caps, attention sinks and grouped-query heads. Disabling it keeps the generated
    /// flash-attention kernels.
    /// </summary>
    public const string NativeAttention = "native_attention";

    /// <summary>
    /// A step whose launches repeat (the same kernels, grids and arguments, e.g. every decode step of a sequence) is recorded once
    /// into a device-side launch graph and replayed with one submission (CUDA graphs), instead of one host call per launch. Like
    /// <see cref="KernelBinaries"/> it does not change generated code, so it is not part of <see cref="DeviceInfo.Features"/>;
    /// disabling it launches every kernel from the host.
    /// </summary>
    public const string LaunchGraphs = "launch_graphs";

    /// <summary>
    /// A kernel may be launched before the previous one in its stream has finished: it waits for that kernel's completion and memory
    /// itself at its start, so the launch and the setup of its blocks overlap the previous kernel's last blocks (CUDA: programmatic
    /// dependent launch, compute capability 9.0 and later). Like <see cref="LaunchGraphs"/> it is not part of
    /// <see cref="DeviceInfo.Features"/>; disabling it launches every kernel after the previous one has finished.
    /// </summary>
    public const string DependentLaunches = "dependent_launches";

    /// <summary>The device reports a live memory budget / usage that accounts for other processes (Vulkan: VK_EXT_memory_budget); without it memory figures are estimates from our own allocations.</summary>
    public const string MemoryBudget = "memory_budget";

    /// <summary>
    /// The device copies directly from host memory it imports, such as the mapped pages of a weight file (Vulkan:
    /// VK_EXT_external_memory_host; CUDA page-locks the pages instead): streamed and selected expert weights then need no staging copy on
    /// the host. Without it they go through staging memory, and expert passes are not split onto the device. Like <see cref="MemoryBudget"/>
    /// it does not change generated code.
    /// </summary>
    public const string HostMemoryImport = "host_memory_import";

    /// <summary>
    /// The driver exports and imports the compiled form of individual kernels as validated, content-keyed binaries (Vulkan:
    /// VK_KHR_pipeline_binary; Metal: binary archives), used for the device code of precompiled models
    /// (<see cref="IExecutionProvider.CompileDeviceSpecificAsync"/>). Without it a provider falls back to a whole-device cache blob
    /// (Vulkan: the pipeline cache). Like <see cref="MemoryBudget"/> it does not change generated code, so it is not part of
    /// <see cref="DeviceInfo.Features"/> or device profiles.
    /// </summary>
    public const string KernelBinaries = "kernel_binaries";

    /// <summary>
    /// Convolutional networks keep their activations channels-last (NHWC) from one convolution, pooling or elementwise op to
    /// the next, with layout conversions only where other ops read them (CPU: hosts with 256-bit or wider SIMD).
    /// </summary>
    public const string ChannelsLast = "channels_last";

    /// <summary>
    /// Paged KV cache for multi-sequence serving (block tables read by the cache-write and attention kernels) with prefix caching by
    /// block content. Detected per device from the support of the paged kernel forms; disabling it keeps the contiguous per-slot
    /// cache. Like <see cref="MemoryBudget"/> it is not part of <see cref="DeviceInfo.Features"/> (it does not change generated code).
    /// </summary>
    public const string PagedKv = "paged_kv";

    /// <summary>
    /// Window-sized KV caches with the paged cache: caches that only sliding-window attention reads keep the blocks a slot's window still
    /// reaches, in a pool of their own (<see cref="Graph.MultiSequence.WindowedCaches"/>). Disabling it gives them a row per position of the
    /// pool like every other cache. Not part of <see cref="DeviceInfo.Features"/> (it does not change generated code).
    /// </summary>
    public const string WindowKv = "window_kv";

    /// <summary>Every feature name defined here.</summary>
    public static IReadOnlyList<string> All { get; } = [MemoryBudget, HostMemoryImport, KernelBinaries, ChannelsLast, PagedKv, WindowKv, MatrixF16, MatrixF16Small, MatrixBF16, MatrixInt8, MatrixFp8, MatrixWorkgroup, MatrixMmaLayout, Float16, BFloat16, Float8, IntDot, SubgroupOps, SubgroupSizeControl, LibraryGemm, NativeGemmFp8, NativeGemmF16, NativeGemmMoe, NativeAttention, TensorCopies, LaunchGraphs, DependentLaunches];

    /// <summary>
    /// Features disabled process-wide through the environment variable <c>SHARPLIGHT_DISABLE_FEATURES</c>
    /// (comma-separated names, or "all"), applied on top of the per-device/per-compile options.
    /// </summary>
    public static IReadOnlySet<string> DisabledByEnvironment { get; } = ParseEnv();

    /// <summary>Effective feature set: detected minus every disable list.</summary>
    public static IReadOnlySet<string> Effective(IReadOnlySet<string> detected, params IEnumerable<string>?[] disabled)
    {
        var set = new HashSet<string>(detected, StringComparer.OrdinalIgnoreCase);
        set.ExceptWith(DisabledByEnvironment);
        foreach (var d in disabled)
        {
            if (d is not null)
            {
                set.ExceptWith(d);
            }
        }

        return set;
    }

    private static HashSet<string> ParseEnv()
    {
        var v = Environment.GetEnvironmentVariable("SHARPLIGHT_DISABLE_FEATURES");
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (string.IsNullOrWhiteSpace(v))
        {
            return set;
        }

        foreach (var f in v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (f.Equals("all", StringComparison.OrdinalIgnoreCase))
            {
                set.UnionWith(All);
            }
            else
            {
                set.Add(f);
            }
        }

        return set;
    }
}
