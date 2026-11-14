using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>qwen3vl</c> / <c>qwen3vlmoe</c> (Qwen3-VL language models) → canonical <c>decoder</c>: Qwen3 (or Qwen3-MoE) with
/// interleaved M-RoPE (<c>rope.sections</c> from <c>rope.dimension_sections</c>). <c>qwen2vl</c> (Qwen2-VL and Qwen2.5-VL language models) is
/// Qwen2 with the same sections, not interleaved (each section a contiguous run of rotary pairs); its projector is not supported, so it runs text-only. Images come from the projector file (an auxiliary
/// <c>clip</c> source with DeepStack layers, see <c>ClipGgufMapping</c>); the DeepStack blocks are added after the first layers.
/// </summary>
internal sealed class Qwen3VlGgufMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    private readonly DecoderGgufMapping _text = new();

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["qwen3vl", "qwen3vlmoe", "qwen2vl"];

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

        if (gguf.Raw("rope.dimension_sections") is Array sections)
        {
            var axes = new int[4];
            for (int i = 0; i < Math.Min(axes.Length, sections.Length); i++)
            {
                axes[i] = Convert.ToInt32(sections.GetValue(i));
            }

            md[Architecture + ".rope.sections"] = axes;
            md[Architecture + ".rope.sections_interleaved"] = gguf.GgufArchitecture != "qwen2vl";
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
