using System.Text.Json;

namespace OnnxBench;

/// <summary>
/// A reference manifest (the ZooTests fixture format): the model, its inputs with shapes (-1 = batch), reference output
/// names, tolerances, and <c>&lt;manifest&gt;.b&lt;batch&gt;.in&lt;k&gt;.bin</c> / <c>.out&lt;k&gt;.bin</c> f32 files next to it.
/// </summary>
internal sealed class ReferenceManifest
{
    public required string Name { get; init; }
    public required string ModelPath { get; init; }
    public required string DataPrefix { get; init; }
    public required int[] Batches { get; init; }
    public required (string Name, long[] Shape)[] Inputs { get; init; }
    public required string[] Outputs { get; init; }
    public double RelativeTolerance { get; init; } = 1e-3;
    public double AbsoluteTolerance { get; init; } = 2e-4;
    /// <summary>When set, parity is judged by relative RMS only.</summary>
    public double? RmsTolerance { get; init; }
    /// <summary>The batch dimension must be fixed at import (reshapes across the batch).</summary>
    public bool FixBatch { get; init; }
    /// <summary>Compare only column 4 of 6-wide detection rows (near-tied detections reorder).</summary>
    public bool ScoresOnly { get; init; }
    public Dictionary<string, double[]> Constants { get; init; } = [];

    public static ReferenceManifest Load(string manifestPath, string modelRoot)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var root = doc.RootElement;
        string name = Path.GetFileNameWithoutExtension(manifestPath);
        string model = root.TryGetProperty("model", out var m) ? m.GetString()! : name + ".onnx";
        bool scoresMode = root.TryGetProperty("mode", out var md) && md.ValueKind == JsonValueKind.String && md.GetString() == "scores6";
        bool fixBatch = root.TryGetProperty("fixbatch", out var f) && f.ValueKind == JsonValueKind.True;
        return new ReferenceManifest
        {
            Name = name,
            ModelPath = Path.Combine(modelRoot, model),
            DataPrefix = manifestPath[..^".json".Length],
            Batches = root.TryGetProperty("batches", out var b) ? b.EnumerateArray().Select(x => x.GetInt32()).ToArray() : [1],
            Inputs = root.GetProperty("inputs").EnumerateArray()
                .Select(e => (e.GetProperty("name").GetString()!, e.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray()))
                .ToArray(),
            Outputs = root.GetProperty("outputs").EnumerateArray().Select(e => e.GetProperty("name").GetString()!).ToArray(),
            RelativeTolerance = Number(root, "rtol") ?? 1e-3,
            AbsoluteTolerance = Number(root, "atol") ?? 2e-4,
            RmsTolerance = Number(root, "rms"),
            FixBatch = fixBatch || name.StartsWith("whisper", StringComparison.Ordinal),
            ScoresOnly = scoresMode || name == "yolov10n",
            Constants = root.TryGetProperty("constants", out var c) && c.ValueKind == JsonValueKind.Object
                ? c.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray())
                : [],
        };
    }

    /// <summary>Reads reference file <paramref name="kind"/> (<c>in</c> or <c>out</c>) number <paramref name="index"/>,
    /// tiling the batch-1 data when there is no file for <paramref name="batch"/>; null when neither exists.</summary>
    public float[]? ReadData(string kind, int index, int batch)
    {
        string path = $"{DataPrefix}.b{batch}.{kind}{index}.bin";
        if (File.Exists(path))
        {
            return TensorCodec.ReadF32(path);
        }

        string single = $"{DataPrefix}.b1.{kind}{index}.bin";
        if (!File.Exists(single))
        {
            return null;
        }

        var one = TensorCodec.ReadF32(single);
        var tiled = new float[one.Length * batch];
        for (int i = 0; i < batch; i++)
        {
            one.CopyTo(tiled, i * one.Length);
        }

        return tiled;
    }

    private static double? Number(JsonElement root, string name) =>
        root.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
}
