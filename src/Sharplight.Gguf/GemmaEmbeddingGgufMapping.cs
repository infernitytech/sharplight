namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>gemma-embedding</c> (EmbeddingGemma): Gemma 3 layers (q/k norms, post-attention and post-MLP norms, GeGLU, embeddings scaled by
/// sqrt(hidden), local layers with their own rope base) with bidirectional attention. Five of every six layers attend within a symmetric window
/// of <c>attention.sliding_window</c> keys. The optional sentence-transformers Dense modules (<c>dense_2</c>, <c>dense_3</c>) become the canonical
/// pooled projections <c>embedding_projection.0</c> / <c>.1</c>.
/// </summary>
internal sealed class GemmaEmbeddingGgufMapping : GgufDecoderFamilyMapping
{
    private const int DefaultPattern = 6;

    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["dense_2.weight"] = "embedding_projection.0.weight",
        ["dense_2.bias"] = "embedding_projection.0.bias",
        ["dense_3.weight"] = "embedding_projection.1.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["gemma-embedding"];

    /// <inheritdoc/>
    protected override string Activation => "gelu_tanh";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? GlobalTensorOverrides => s_global;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        const string a = GgufDecoderHyperParameters.Prefix;
        int layers = TrunkLayerCount(g);
        md[a + "attention.causal"] = false;
        md[a + "embedding_scale"] = MathF.Sqrt(g.Int("embedding_length"));
        md[a + "attention.scale"] = 1f / MathF.Sqrt(HeadDimOf(g));
        md[a + "attention.sliding_window"] = GgufMappingHelpers.SymmetricWindow(g.Int("attention.sliding_window"));
        md[a + "attention.sliding_layers"] = GgufMappingHelpers.SlidingLayers(g, layers, DefaultPattern, denseFirst: false);
        md[a + "rope.local_base"] = g.Float("rope.freq_base_swa", GgufMappingHelpers.DefaultRopeBase);
    }
}
