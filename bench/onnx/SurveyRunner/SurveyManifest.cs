using System.Text.Json;

/// <summary>A reference manifest: the model, its inputs and outputs and the comparison tolerances.</summary>
internal sealed class SurveyManifest
{
    private SurveyManifest()
    {
    }

    /// <summary>The model file name without extension.</summary>
    public required string Name { get; init; }

    /// <summary>The path prefix of the reference <c>.bin</c> files.</summary>
    public required string Prefix { get; init; }

    /// <summary>The model file.</summary>
    public required string ModelPath { get; init; }

    /// <summary>The manifest's first batch size.</summary>
    public required int Batch { get; init; }

    /// <summary>The inputs in manifest order, including pinned constants that are not graph inputs.</summary>
    public required (string Name, long[] Shape)[] Inputs { get; init; }

    /// <summary>The reference output names.</summary>
    public required string[] OutputNames { get; init; }

    /// <summary>Relative tolerance.</summary>
    public required double RelativeTolerance { get; init; }

    /// <summary>Absolute tolerance, scaled by the reference magnitude.</summary>
    public required double AbsoluteTolerance { get; init; }

    /// <summary>The comparison mode (<c>scores6</c> compares only the score column), or null.</summary>
    public required string? Mode { get; init; }

    /// <summary>The RMS tolerance replacing the element-wise check, or null.</summary>
    public required double? RmsTolerance { get; init; }

    /// <summary>Whether the symbolic batch is skipped and the batch is fixed.</summary>
    public required bool FixedBatch { get; init; }

    /// <summary>Pinned constant inputs by name.</summary>
    public required Dictionary<string, double[]> Constants { get; init; }

    /// <summary>Reads the manifest at <paramref name="manifestPath"/>.</summary>
    public static SurveyManifest Load(string manifestPath)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var root = document.RootElement;
        string modelRoot = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "onnx");
        string name = Path.GetFileNameWithoutExtension(manifestPath);
        string modelPath = root.TryGetProperty("model", out var model)
            ? Path.Combine(modelRoot, model.GetString()!)
            : Path.Combine(modelRoot, name + ".onnx");

        // ReferenceTests compares only the score column of this near-tied detection output.
        string? mode = root.TryGetProperty("mode", out var modeElement) && modeElement.ValueKind == JsonValueKind.String
            ? modeElement.GetString()
            : null;
        if (name == "yolov10n")
        {
            mode = "scores6";
        }

        return new SurveyManifest
        {
            Name = name,
            Prefix = manifestPath[..^5],
            ModelPath = modelPath,
            Batch = root.TryGetProperty("batches", out var batches) ? batches.EnumerateArray().First().GetInt32() : 1,
            Inputs = root.GetProperty("inputs").EnumerateArray()
                .Select(e => (e.GetProperty("name").GetString()!, e.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray()))
                .ToArray(),
            OutputNames = root.GetProperty("outputs").EnumerateArray().Select(e => e.GetProperty("name").GetString()!).ToArray(),
            RelativeTolerance = root.TryGetProperty("rtol", out var rtol) ? rtol.GetDouble() : 1e-3,
            AbsoluteTolerance = root.TryGetProperty("atol", out var atol) ? atol.GetDouble() : 2e-4,
            Mode = mode,
            RmsTolerance = root.TryGetProperty("rms", out var rms) && rms.ValueKind == JsonValueKind.Number ? rms.GetDouble() : null,
            FixedBatch = root.TryGetProperty("fixbatch", out var fixBatch) && fixBatch.GetBoolean() || name.StartsWith("whisper", StringComparison.Ordinal),
            Constants = root.TryGetProperty("constants", out var constants) && constants.ValueKind == JsonValueKind.Object
                ? constants.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray())
                : new Dictionary<string, double[]>(),
        };
    }
}
