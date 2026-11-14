namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>modern-bert</c> (ModernBERT, Granite Embedding R2): a bidirectional pre-norm encoder with bias-free layer norms (one after the
/// token embedding, none in front of the first attention), a fused q/k/v projection, rotary embeddings and a fused GLU up projection (activated
/// half first; GELU, or SiLU when <c>hidden_act</c> says so). With <c>attention.sliding_window</c> every <c>attention.sliding_window_pattern</c>-th
/// layer (the first included) is global and the others attend within a symmetric window with the local rope base. Classifier tensors are kept
/// under <c>classifier.*</c> and not used for embeddings.
/// </summary>
internal sealed class ModernBertGgufMapping : GgufDecoderFamilyMapping
{
    private const int DefaultPattern = 3;

    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["cls.weight"] = "classifier.pre.weight",
        ["cls.bias"] = "classifier.pre.bias",
        ["cls.norm.weight"] = "classifier.norm.weight",
        ["cls.output.weight"] = "classifier.out.weight",
        ["cls.output.bias"] = "classifier.out.bias",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["modern-bert"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? GlobalTensorOverrides => s_global;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        const string a = GgufDecoderHyperParameters.Prefix;
        md[a + "attention.causal"] = false;
        md[a + "mlp.activation"] = (g.Raw("hidden_act") as string)?.ToLowerInvariant() switch
        {
            null or "gelu" or "geglu" => "gelu",
            "gelu_pytorch_tanh" or "gelu_new" => "gelu_tanh",
            "silu" or "swish" or "swiglu" => "silu",
            var other => throw new NotSupportedException($"GGUF modern-bert: hidden_act '{other}' is not supported."),
        };
        if (g.Int("attention.sliding_window", 0) is > 0 and var window)
        {
            md[a + "attention.sliding_window"] = GgufMappingHelpers.SymmetricWindow(window);
            md[a + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(g, TrunkLayerCount(g), DefaultPattern, denseFirst: true);
            md[a + "rope.local_base"] = g.Float("rope.freq_base_swa", GgufMappingHelpers.DefaultRopeBase);
        }
    }
}
