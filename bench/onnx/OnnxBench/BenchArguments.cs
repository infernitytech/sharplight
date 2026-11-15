using System.Globalization;

namespace OnnxBench;

/// <summary>Command line of the benchmark: <c>OnnxBench &lt;manifest.json&gt; &lt;cpu|vulkan&gt; [options]</c>.</summary>
internal sealed class BenchArguments
{
    public const string Usage =
        "usage: OnnxBench <manifest.json> <cpu|vulkan> [--runs N] [--budget-s S] [--batch B] [--device D] [--profile] [--reduced-precision] [--fast-precision] [--no-split-f16] [--no-fuse-projections] [--vary-input]\n" +
        "  the \"model\" entry of the manifest resolves against $SHARPLIGHT_ONNX_MODELS.";

    public required string ManifestPath { get; init; }
    public required string Provider { get; init; }
    /// <summary>Most timed runs (after one untimed warm-up run).</summary>
    public int Runs { get; init; } = 30;
    /// <summary>Wall-clock budget for the timed runs; at least 5 runs are always made.</summary>
    public double BudgetSeconds { get; init; } = 6;
    /// <summary>Batch size; when the manifest has no reference data for it, the batch-1 data is tiled.</summary>
    public int? Batch { get; init; }
    public int Device { get; init; }
    /// <summary>Print the per-kernel GPU profile (Vulkan) after timing.</summary>
    public bool Profile { get; init; }
    public bool ReducedPrecision { get; init; }
    /// <summary>One f16 product of row-scaled inputs per multiply-add in f32 matmuls, convs and attention (CompileOptions.MatrixPrecision = Fast, TF32's accuracy class).</summary>
    public bool FastPrecision { get; init; }
    /// <summary>f32 matmuls may run on the split-f16 tensor-core GEMM (CompileOptions.SplitHalfMatMul, default; --no-split-f16 keeps the SIMT GEMM).</summary>
    public bool SplitHalf { get; init; }

    /// <summary>Keeps sibling projections separate (OnnxImportOptions.FuseProjections = false).</summary>
    public bool NoProjectionFusion { get; init; }

    /// <summary>
    /// Scales every f32 input by a new random factor per element before each timed run (outside the timing), so that data-dependent
    /// counts change from run to run as with new images (parity is checked on the reference input first).
    /// </summary>
    public bool VaryInput { get; init; }

    public static BenchArguments? Parse(string[] args)
    {
        if (args.Length < 2 || args[1] is not ("cpu" or "vulkan"))
        {
            return null;
        }

        var options = args.Skip(2).ToList();
        string? Value(string name)
        {
            int i = options.IndexOf(name);
            return i >= 0 && i + 1 < options.Count ? options[i + 1] : null;
        }

        return new BenchArguments
        {
            ManifestPath = args[0],
            Provider = args[1],
            Runs = Value("--runs") is { } runs ? int.Parse(runs, CultureInfo.InvariantCulture) : 30,
            BudgetSeconds = Value("--budget-s") is { } budget ? double.Parse(budget, CultureInfo.InvariantCulture) : 6,
            Batch = Value("--batch") is { } batch ? int.Parse(batch, CultureInfo.InvariantCulture) : null,
            Device = Value("--device") is { } device ? int.Parse(device, CultureInfo.InvariantCulture) : 0,
            Profile = options.Contains("--profile"),
            ReducedPrecision = options.Contains("--reduced-precision"),
            FastPrecision = options.Contains("--fast-precision"),
            SplitHalf = !options.Contains("--no-split-f16"),
            NoProjectionFusion = options.Contains("--no-fuse-projections"),
            VaryInput = options.Contains("--vary-input"),
        };
    }
}
