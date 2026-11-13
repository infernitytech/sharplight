using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>qwen3vl</c> / <c>qwen3vlmoe</c> (Qwen3-VL language models) → canonical <c>decoder</c>, as text-only models.
/// The text model is Qwen3 with interleaved M-RoPE and DeepStack: for text tokens every M-RoPE axis carries the same
/// position, which is exactly plain (neox) RoPE, and DeepStack only adds vision features to image positions, so the
/// dense decoder reproduces it. Image input (M-RoPE image positions, DeepStack injection) is not supported yet.
/// </summary>
internal sealed class Qwen3VlGgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    private readonly DecoderGgufMapping _text = new();

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["qwen3vl", "qwen3vlmoe"];

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView gguf)
    {
        var md = new Dictionary<string, object>(_text.MapHyperParameters(gguf), StringComparer.Ordinal);
        if (gguf.Has("expert_count"))
        {
            string a = Architecture + ".";
            md[a + "moe.expert_count"] = gguf.Int("expert_count");
            md[a + "moe.expert_used_count"] = gguf.Int("expert_used_count");
            md[a + "moe.ffn_hidden_size"] = gguf.Int("expert_feed_forward_length");
            md[a + "moe.normalize_weights"] = true;
        }

        return md;
    }

    /// <inheritdoc/>
    public string? MapWeightName(string ggufName, GgufMetadataView gguf) => _text.MapWeightName(ggufName, gguf);

    /// <inheritdoc/>
    public WeightInfo MapWeight(WeightInfo weight, GgufMetadataView gguf) => _text.MapWeight(weight, gguf);

    /// <inheritdoc/>
    public IEnumerable<WeightInfo> SyntheticWeights(GgufMetadataView gguf, IReadOnlyDictionary<string, WeightInfo> weights)
        => _text.SyntheticWeights(gguf, weights);

    /// <inheritdoc/>
    public IReadOnlyList<WeightInfo>? Split(string ggufName, WeightInfo tensor, GgufMetadataView gguf) => _text.Split(ggufName, tensor, gguf);
}
