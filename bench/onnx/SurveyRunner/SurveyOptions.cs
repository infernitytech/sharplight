using System.Globalization;

/// <summary>The command line of the survey harness.</summary>
internal sealed class SurveyOptions
{
    private readonly List<string> _arguments;

    private SurveyOptions(List<string> arguments)
    {
        _arguments = arguments;
        ManifestPath = arguments[0];
        ProviderName = arguments[1];
        MaxRuns = int.Parse(Value("--runs") ?? "30", CultureInfo.InvariantCulture);
        BudgetSeconds = double.Parse(Value("--budget-s") ?? "4", CultureInfo.InvariantCulture);
        DeviceIndex = int.Parse(Value("--device") ?? "0", CultureInfo.InvariantCulture);
        BatchOverride = Value("--batch") is { } batch ? int.Parse(batch, CultureInfo.InvariantCulture) : null;
        DumpDirectory = Value("--dump");
        ReducedPrecision = arguments.Contains("--reduced-precision");
        Profile = arguments.Contains("--profile");
    }

    /// <summary>The manifest JSON file.</summary>
    public string ManifestPath { get; }

    /// <summary><c>cpu</c>, <c>vulkan</c> or <c>cuda</c>.</summary>
    public string ProviderName { get; }

    /// <summary>The most timed runs.</summary>
    public int MaxRuns { get; }

    /// <summary>The timing budget in seconds (at least three runs are always timed).</summary>
    public double BudgetSeconds { get; }

    /// <summary>Index of the device to open.</summary>
    public int DeviceIndex { get; }

    /// <summary>The batch size overriding the manifest's first batch, or null.</summary>
    public int? BatchOverride { get; }

    /// <summary>The directory receiving every graph output as raw f32, or null.</summary>
    public string? DumpDirectory { get; }

    /// <summary>Whether reduced-precision kernels are allowed.</summary>
    public bool ReducedPrecision { get; }

    /// <summary>Whether the GPU segments (Vulkan or CUDA) are profiled after timing.</summary>
    public bool Profile { get; }

    /// <summary>Parses <paramref name="args"/>.</summary>
    public static SurveyOptions Parse(string[] args)
    {
        return new SurveyOptions(args.ToList());
    }

    /// <summary>The <c>--bound name=value</c> pairs (runtime values that size a data-dependent extent).</summary>
    public IEnumerable<KeyValuePair<string, long>> Bounds()
    {
        for (int i = 0; i + 1 < _arguments.Count; i++)
        {
            if (_arguments[i] != "--bound")
            {
                continue;
            }

            string[] pair = _arguments[i + 1].Split('=');
            yield return new KeyValuePair<string, long>(pair[0], long.Parse(pair[1], CultureInfo.InvariantCulture));
        }
    }

    private string? Value(string name)
    {
        int index = _arguments.IndexOf(name);
        return index >= 0 && index + 1 < _arguments.Count ? _arguments[index + 1] : null;
    }
}
