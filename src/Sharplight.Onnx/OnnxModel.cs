namespace Sharplight.Onnx;

internal sealed class OnnxModel
{
    public long IrVersion { get; init; }
    public string ProducerName { get; init; } = "";
    public string ProducerVersion { get; init; } = "";
    public string Domain { get; init; } = "";
    public long ModelVersion { get; init; }
    public string DocString { get; init; } = "";
    /// <summary>Opset versions per domain ("" = the default ai.onnx domain).</summary>
    public IReadOnlyDictionary<string, long> Opsets { get; init; } = new Dictionary<string, long>();
    public IReadOnlyDictionary<string, string> MetadataProps { get; init; } = new Dictionary<string, string>();
    public required OnnxGraph Graph { get; init; }
    /// <summary>Options consulted by <see cref="OnnxGraphAdapter"/> (input shape overrides, ...).</summary>
    public OnnxImportOptions Options { get; set; } = new();

    public long Opset(string domain = "") => Opsets.TryGetValue(domain, out var v) ? v
        : domain == "" && Opsets.TryGetValue("ai.onnx", out v) ? v : 0;
}
