namespace Sharplight.Onnx;

internal sealed class OnnxNode
{
    public string Name { get; init; } = "";
    public required string OpType { get; init; }
    public string Domain { get; init; } = "";
    /// <summary>Input names; empty string marks an omitted optional input.</summary>
    public required IReadOnlyList<string> Inputs { get; init; }
    public required IReadOnlyList<string> Outputs { get; init; }
    public IReadOnlyDictionary<string, OnnxAttribute> Attributes { get; init; } = new Dictionary<string, OnnxAttribute>();

    public bool Has(string a) => Attributes.ContainsKey(a);
    public long Int(string a, long def) => Attributes.TryGetValue(a, out var x) ? x.I : def;
    public float Float(string a, float def) => Attributes.TryGetValue(a, out var x) ? x.F : def;
    public string Str(string a, string def) => Attributes.TryGetValue(a, out var x) && x.S is not null ? x.S : def;
    public long[]? Ints(string a) => Attributes.TryGetValue(a, out var x) ? [.. x.Ints] : null;
    public float[]? Floats(string a) => Attributes.TryGetValue(a, out var x) ? [.. x.Floats] : null;
    public OnnxTensor? Tensor(string a) => Attributes.TryGetValue(a, out var x) ? x.T : null;

    public override string ToString() => $"{OpType} {Name} ({string.Join(", ", Inputs)}) -> ({string.Join(", ", Outputs)})";
}
