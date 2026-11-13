using System.Text.Json;

namespace Sharplight.Onnx.Tests;

/// <summary>A reference case produced by .work/onnx_ref/*.py: model + manifest + inputs/outputs from onnxruntime.</summary>
internal sealed class RefCase
{
    public required string Name { get; init; }
    public required string ModelPath { get; init; }
    public required string Prefix { get; init; }
    public required int[] Batches { get; init; }
    public required RefInput[] Inputs { get; init; }
    public required string[] Outputs { get; init; }
    public double Rtol { get; init; } = 1e-3;
    public double Atol { get; init; } = 2e-4;
    /// <summary>Import with the concrete batch size instead of the symbolic <c>N</c> (models that reshape/unsqueeze across the batch dim).</summary>
    public bool FixBatch { get; init; }
    /// <summary>Relative RMS error criterion for noisy cases (quantized models) instead of the element-wise tolerance; null = element-wise.</summary>
    public double? RmsTol { get; init; }
    /// <summary>Comparison mode hint: "scores6" = detections as rows of [x1,y1,x2,y2,score,class]; only the score column is compared (tied/near-tied rows swap).</summary>
    public string? Mode { get; init; }
    /// <summary>Inputs pinned to constants at import (manifest "constants": {name: [values]}); they are not fed at run time.</summary>
    public Dictionary<string, double[]> Constants { get; init; } = new();
    public IEnumerable<int> RuntimeInputs => Enumerable.Range(0, Inputs.Length).Where(k => !Constants.ContainsKey(Inputs[k].Name));

    public static string RefRoot => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "onnx", "ref");
    public static string SynthRoot => Path.Combine(RefRoot, "synth");

    public static RefCase? TryLoad(string manifestPath, string modelPath)
    {
        if (!File.Exists(manifestPath) || !File.Exists(modelPath))
        {
            return null;
        }

        using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var r = doc.RootElement;
        return new RefCase
        {
            Name = Path.GetFileNameWithoutExtension(manifestPath),
            ModelPath = modelPath,
            Prefix = manifestPath[..^".json".Length],
            Batches = r.GetProperty("batches").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            Inputs = r.GetProperty("inputs").EnumerateArray().Select(e => new RefInput(e.GetProperty("name").GetString()!, e.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray())).ToArray(),
            Outputs = r.GetProperty("outputs").EnumerateArray().Select(e => e.GetProperty("name").GetString()!).ToArray(),
            Rtol = r.TryGetProperty("rtol", out var rt) ? rt.GetDouble() : 1e-3,
            Atol = r.TryGetProperty("atol", out var at) ? at.GetDouble() : 2e-4,
            FixBatch = r.TryGetProperty("fixbatch", out var fb) && fb.GetBoolean(),
            Constants = r.TryGetProperty("constants", out var cs) && cs.ValueKind == JsonValueKind.Object
                ? cs.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(x => x.GetDouble()).ToArray()) : new(),
            Mode = r.TryGetProperty("mode", out var md) && md.ValueKind == JsonValueKind.String ? md.GetString() : null,
            RmsTol = r.TryGetProperty("rms", out var rm) && rm.ValueKind == JsonValueKind.Number ? rm.GetDouble() : null,
        };
    }

    public static RefCase? Synthetic(string name) => TryLoad(Path.Combine(SynthRoot, name + ".json"), Path.Combine(SynthRoot, name + ".onnx"));
    public static RefCase? Model(string name, string? onnxFile = null) => TryLoad(Path.Combine(RefRoot, name + ".json"), Path.Combine(Path.GetDirectoryName(RefRoot)!, (onnxFile ?? name) + ".onnx"));

    /// <summary>Zoo models (scripts/make_zoo_refs.py): manifest in ref/zoo carries the model path relative to ~/models/onnx.</summary>
    public static string ZooRoot => Path.Combine(RefRoot, "zoo");
    public static RefCase? Zoo(string name)
    {
        var manifest = Path.Combine(ZooRoot, name + ".json");
        if (!File.Exists(manifest))
        {
            return null;
        }

        using var doc = JsonDocument.Parse(File.ReadAllText(manifest));
        return TryLoad(manifest, Path.Combine(Path.GetDirectoryName(RefRoot)!, doc.RootElement.GetProperty("model").GetString()!));
    }
    public static IEnumerable<string> ZooNames()
        => Directory.Exists(ZooRoot) ? Directory.GetFiles(ZooRoot, "*.json").Select(f => Path.GetFileNameWithoutExtension(f)!).OrderBy(x => x) : [];

    public static IEnumerable<string> SyntheticNames()
        => Directory.Exists(SynthRoot) ? Directory.GetFiles(SynthRoot, "*.json").Select(f => Path.GetFileNameWithoutExtension(f)!).OrderBy(x => x) : [];

    public float[] ReadInput(int batch, int k) => ReadF32($"{Prefix}.b{batch}.in{k}.bin");
    public float[] ReadOutput(int batch, int k) => ReadF32($"{Prefix}.b{batch}.out{k}.bin");

    private static float[] ReadF32(string path)
    {
        var b = File.ReadAllBytes(path);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }
}
