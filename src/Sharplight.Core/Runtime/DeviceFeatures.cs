namespace Sharplight.Runtime;

/// <summary>
/// Provider-neutral names of optional device features. A provider maps its own mechanisms onto these
/// (e.g. Vulkan VK_KHR_cooperative_matrix, CUDA tensor cores / mma.sync, Metal simdgroup_matrix all
/// provide <see cref="MatrixF16"/>). Hosts can turn any of them off for a device or a single compilation.
/// </summary>
/// <remarks>
/// Diagnostic environment switches of Sharplight.Core:
/// <list type="bullet">
/// <item><c>SHARPLIGHT_DISABLE_FEATURES</c>: comma-separated feature names (or "all") disabled for every device, see <see cref="DisabledByEnvironment"/>.</item>
/// <item><c>SHARPLIGHT_SUBMISSION_BUDGET_MS</c>: process-wide override of <see cref="DeviceInfo.SubmissionBudgetMs"/>; read by the providers.</item>
/// </list>
/// </remarks>
public static class DeviceFeatures
{
    /// <summary>Subgroup/warp-level matrix multiply-accumulate on f16 inputs with f32 accumulation (tensor cores).</summary>
    public const string MatrixF16 = "matrix_f16";
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

    /// <summary>The device reports a live memory budget / usage that accounts for other processes (Vulkan: VK_EXT_memory_budget); without it memory figures are estimates from our own allocations.</summary>
    public const string MemoryBudget = "memory_budget";

    /// <summary>Every feature name defined here.</summary>
    public static IReadOnlyList<string> All { get; } = [MemoryBudget, MatrixF16, MatrixBF16, MatrixInt8, MatrixFp8, MatrixWorkgroup, MatrixMmaLayout, Float16, BFloat16, Float8, IntDot, SubgroupOps, SubgroupSizeControl];

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
