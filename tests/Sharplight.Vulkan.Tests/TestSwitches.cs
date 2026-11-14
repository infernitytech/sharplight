namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Every <c>SHARPLIGHT_*</c> environment switch read by the Vulkan tests (the library's own switches are listed in the
/// <c>DiagnosticSwitches</c> classes of the Shaders and Vulkan projects).
/// <list type="bullet">
/// <item><c>SHARPLIGHT_TEST_DEVICE=&lt;index&gt;</c>: run on another Vulkan device (e.g. the iGPU or llvmpipe) to exercise
/// feature fallbacks; the default is the first discrete GPU.</item>
/// <item><c>SHARPLIGHT_BENCH=1</c>: enable the GPU micro-benchmarks of <c>BenchTests</c>.</item>
/// <item><c>SHARPLIGHT_BENCH_TYPES=q4_k,q8_0</c>: run the matmul benchmarks only for these weight types.</item>
/// <item><c>SHARPLIGHT_BENCH_REPORT=1</c>: print the per-kernel profile of every matmul benchmark.</item>
/// <item><c>SHARPLIGHT_BENCH_SHAPES=K,N;K,N</c>: matmul benchmark shapes instead of the default 27B / Mistral set.</item>
/// <item><c>SHARPLIGHT_BENCH_ROWS=n</c>: activation rows of the matmul benchmarks (default 512).</item>
/// </list>
/// </summary>
internal static class TestSwitches
{
    /// <summary>Index of the Vulkan device to test on, or null for the default.</summary>
    public static int? DeviceIndex { get; } = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_DEVICE"), out int index) ? index : null;

    /// <summary>True when benchmarks were requested.</summary>
    public static bool BenchmarksEnabled { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH") == "1";

    /// <summary>Weight types the matmul benchmarks are limited to, or null for all.</summary>
    public static IReadOnlySet<string>? BenchmarkTypes { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_TYPES")?.Split(',').ToHashSet();

    /// <summary>Matmul benchmark shapes (K, N), or null for the default set.</summary>
    public static (int K, int N)[]? BenchmarkShapes { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_SHAPES")?
        .Split(';').Select(shape => shape.Split(',')).Select(kn => (int.Parse(kn[0]), int.Parse(kn[1]))).ToArray();

    /// <summary>Activation rows of the matmul benchmarks.</summary>
    public static int BenchmarkRows { get; } = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_ROWS"), out int rows) ? rows : 512;

    /// <summary>
    /// Weight copies the matmul benchmarks multiply in turn (default 1). With several copies whose total size exceeds the
    /// GPU's last-level cache, every matmul reads its weights from memory, as in a model's step.
    /// </summary>
    public static int BenchmarkWeightCopies { get; } = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_COPIES"), out int copies) ? copies : 1;

    /// <summary>
    /// Profiled runs of each matmul benchmark (default 50; minimum times are reported). More runs give the GPU time to reach
    /// its full clocks, which short benchmarks of fast kernels otherwise miss.
    /// </summary>
    public static int BenchmarkIterations { get; } = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_ITERS"), out int iters) ? iters : 50;

    /// <summary>True when the matmul benchmarks measure gate / up pairs, SiLU(x·Wa)·(x·Wb), instead of single matmuls.</summary>
    public static bool BenchmarkDual { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_DUAL") == "1";

    /// <summary>True when the matmul benchmarks print their per-kernel profiles.</summary>
    public static bool BenchmarkReports { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_REPORT") == "1";
}
