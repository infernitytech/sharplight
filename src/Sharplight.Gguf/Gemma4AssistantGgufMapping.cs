using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>gemma4-assistant</c> (Gemma 4 multi-token-prediction drafters) → canonical <c>shared_kv_draft</c> (vocabulary on
/// <c>SharedKvDraftHead</c>). The layers and their metadata are <c>gemma4</c>'s and are mapped by the <c>gemma4</c> mapping;
/// <c>nextn.pre_projection</c> / <c>nextn.post_projection</c> are the input / output projections and <c>embedding_length_out</c> the target width.
/// The converter drops the centroid head, so the full LM head ranks every token.
/// </summary>
internal sealed class Gemma4AssistantGgufMapping : IGgufArchitectureMapping
{
    private const string LayerArchitecture = "gemma4";
    private static readonly DecoderGgufMapping s_layers = new();

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["gemma4-assistant"];

    /// <inheritdoc/>
    public string Architecture => "shared_kv_draft";

    /// <inheritdoc/>
    public bool IsDraftModel => true;

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string decoderPrefix = s_layers.Architecture + ".";
        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, value) in s_layers.MapHyperParameters(AsLayers(g)))
        {
            md[key.StartsWith(decoderPrefix, StringComparison.Ordinal) ? a + key[decoderPrefix.Length..] : key] = value;
        }

        md.Remove(a + "final_logit_softcap");   // the assistant's head has no soft cap
        // Its sliding layers see the keys within the inclusive radius sliding_window of the query, as in the reference implementation (a flipped
        // bidirectional window): one more than the canonical (exclusive) window. llama.cpp's graph sees one key less than the window.
        if (md.TryGetValue(a + "attention.sliding_window", out var window) && window is int w && w > 0)
        {
            md[a + "attention.sliding_window"] = w + 1;
        }

        md[a + "target_hidden_size"] = g.Int("embedding_length_out");
        return md;
    }

    /// <inheritdoc/>
    public string? MapWeightName(string ggufName, GgufMetadataView g) => ggufName switch
    {
        "nextn.pre_projection.weight" => "input_projection.weight",
        "nextn.post_projection.weight" => "output_projection.weight",
        _ => s_layers.MapWeightName(ggufName, AsLayers(g)),
    };

    /// <inheritdoc/>
    public WeightInfo MapWeight(WeightInfo weight, GgufMetadataView g) => s_layers.MapWeight(weight, AsLayers(g));

    /// <inheritdoc/>
    public IEnumerable<WeightInfo> SyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
        => s_layers.SyntheticWeights(AsLayers(g), weights);

    // The file's metadata seen as a gemma4 model's (gemma4-assistant.* keys as gemma4.*).
    private static GgufMetadataView AsLayers(GgufMetadataView g)
    {
        string prefix = g.GgufArchitecture + ".";
        var renamed = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, value) in g.All)
        {
            renamed[key.StartsWith(prefix, StringComparison.Ordinal) ? LayerArchitecture + "." + key[prefix.Length..] : key] = value;
        }

        return new GgufMetadataView(renamed, LayerArchitecture, g.Tensor);
    }
}
