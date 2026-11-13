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

    /// <summary>True when the matmul benchmarks print their per-kernel profiles.</summary>
    public static bool BenchmarkReports { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_BENCH_REPORT") == "1";
}
