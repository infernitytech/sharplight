using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>minimax-m3</c> (MiniMax-M3): llama layers with per-head q/k RMS norms (the converter bakes the Gemma +1 into all norms), partial
/// rope, leading dense layers and routed-expert layers with a shared expert, everything with llama.cpp's clamped SwiGLU-OAI (alpha 1.702,
/// limit 7). MiniMax sparse attention: every query sees all key blocks while it sees at most <c>attention.indexer.top_k</c> blocks of
/// <c>attention.indexer.block_size</c> keys, so dense attention is exact up to that context; block selection beyond it is not implemented
/// yet, so the context is capped there and the indexer tensors are dropped.
/// </summary>
internal sealed class MiniMaxM3GgufMapping : GgufDecoderFamilyMapping, IGgufWeightTransformer
{
    private const string IndexerPrefix = "attention.block_indexer.";

    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal)
    {
        ["indexer.q_proj.weight"] = IndexerPrefix + "q.weight",
        ["indexer.k_proj.weight"] = IndexerPrefix + "k.weight",
        ["indexer.q_norm.weight"] = IndexerPrefix + "q_norm.weight",
        ["indexer.k_norm.weight"] = IndexerPrefix + "k_norm.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["minimax-m3"];

    /// <inheritdoc/>
    protected override string Activation => "swiglu_oai";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        const string a = GgufDecoderHyperParameters.Prefix;
        md[a + "mlp.swiglu_alpha"] = 1.702f;   // the constants of llama.cpp's LLM_FFN_SWIGLU_OAI_MOE
        md[a + "mlp.swiglu_limit"] = 7f;
        md[a + "moe.normalize_weights"] = true;
        if (g.Int("attention.indexer.top_k", 0) is > 0 and var blocks)
        {
            int context = blocks * g.Int("attention.indexer.block_size", 1);
            md[a + "context_length"] = md.TryGetValue(a + "context_length", out var current) && current is int known ? Math.Min(known, context) : context;
        }
    }

    /// <summary>Drops the indexer tensors, which dense attention does not need.</summary>
    public void Transform(Dictionary<string, WeightInfo> weights, GgufMetadataView gguf)
    {
        foreach (var name in weights.Keys.Where(n => n.Contains(IndexerPrefix, StringComparison.Ordinal)).ToList())
        {
            weights.Remove(name);
        }
    }
}
