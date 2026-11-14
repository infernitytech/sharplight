using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Gemma 4 assistants (<c>gemma4_assistant</c>, <c>gemma4_unified_assistant</c>: multi-token-prediction drafters for Gemma 4 targets) → canonical
/// <c>shared_kv_draft</c> (vocabulary on <c>SharedKvDraftHead</c>). The layers are Gemma 4 text layers that share every key / value with the
/// target, mapped by the <c>gemma4_text</c> mapping under the drafter's architecture; <c>pre_projection</c> / <c>post_projection</c> become the
/// input / output projections, <c>backbone_hidden_size</c> the target width. With <c>use_ordered_embeddings</c> the masked embedder's
/// <c>centroids</c> are the centroid head and its <c>token_ordering</c> (the tokens of cluster c at entries [c·V/C, (c+1)·V/C)) becomes the cluster
/// of every token.
/// </summary>
internal sealed class Gemma4AssistantHfMapping : IHfArchitectureMapping
{
    private const string LayerModelType = "gemma4_text";
    private static readonly DecoderHfMapping s_layers = new();

    /// <inheritdoc/>
    public string Architecture => "shared_kv_draft";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["gemma4_assistant", "gemma4_unified_assistant"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string decoderPrefix = s_layers.Architecture + ".";
        string a = Architecture + ".";
        using var layers = c.AsModelType(LayerModelType);
        var md = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, value) in s_layers.MapHyperParameters(layers, options))
        {
            md[key.StartsWith(decoderPrefix, StringComparison.Ordinal) ? a + key[decoderPrefix.Length..] : key] = value;
        }

        md.Remove(a + "final_logit_softcap");   // the assistant's head has no soft cap
        // Its sliding layers see the keys within the inclusive radius sliding_window of the query (a flipped bidirectional window): one more
        // than the canonical (exclusive) window.
        if (md.TryGetValue(a + "attention.sliding_window", out var window) && window is int w && w > 0)
        {
            md[a + "attention.sliding_window"] = w + 1;
        }

        md[a + "target_hidden_size"] = c.Int("backbone_hidden_size");
        if (c.GetBool("use_ordered_embeddings") == true)
        {
            md[a + "centroid_top_k"] = c.GetInt("centroid_intermediate_top_k") ?? 32;
        }

        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var own = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var layerTensors = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        WeightInfo? ordering = null;
        foreach (var (name, tensor) in tensors)
        {
            switch (name)
            {
                case "pre_projection.weight":
                    own["input_projection.weight"] = tensor.ToWeightInfo().As("input_projection.weight");
                    break;
                case "post_projection.weight":
                    own["output_projection.weight"] = tensor.ToWeightInfo().As("output_projection.weight");
                    break;
                case "masked_embedding.centroids.weight":
                    own["centroids.weight"] = tensor.ToWeightInfo().As("centroids.weight");
                    break;
                case "masked_embedding.token_ordering":
                    ordering = tensor.ToWeightInfo();
                    break;
                default:
                    layerTensors[name] = tensor;
                    break;
            }
        }

        using var layers = c.AsModelType(LayerModelType);
        var weights = new Dictionary<string, WeightInfo>(s_layers.MapWeights(layers, layerTensors, options), StringComparer.Ordinal);
        foreach (var (name, w) in own)
        {
            weights[name] = w;
        }

        if (c.GetBool("use_ordered_embeddings") != true)
        {
            weights.Remove("centroids.weight");
        }
        else if (ordering is not null)
        {
            weights["centroid_of_token"] = CentroidOfToken(ordering, c.GetInt("num_centroids") ?? 2048);
        }

        return weights;
    }

    // token_ordering lists the tokens cluster by cluster (V / C each); the canonical form is the cluster of every token.
    private static WeightInfo CentroidOfToken(WeightInfo ordering, int centroids)
    {
        int vocab = (int)ordering.Shape.Elements();
        int perCentroid = vocab / centroids;
        return WeightInfo.Derived("centroid_of_token", DataType.I32, new Shape(vocab), () =>
        {
            var raw = ordering.ReadAll();
            var bytes = new byte[vocab * 4];
            for (int i = 0; i < vocab; i++)
            {
                long token = ordering.Type == DataType.I64 ? BitConverter.ToInt64(raw, i * 8) : BitConverter.ToInt32(raw, i * 4);
                BitConverter.TryWriteBytes(bytes.AsSpan(checked((int)token) * 4, 4), i / perCentroid);
            }

            return bytes;
        });
    }
}
