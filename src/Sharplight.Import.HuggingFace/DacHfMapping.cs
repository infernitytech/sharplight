using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face DAC (<c>dac</c>: DacModel, the Descript Audio Codec with snake activations and a factorized, cosine-similarity residual quantizer) → canonical <c>dac</c>
/// (documented on <c>DacAdapter</c>). The checkpoint's names are kept (its weight normalization is already folded); the snake parameters <c>[1, C, 1]</c> become <c>[C]</c>.
/// </summary>
internal sealed class DacHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "dac";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["dac"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "sample_rate"] = c.Int("sampling_rate"),
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "encoder_hidden_size"] = c.Int("encoder_hidden_size"),
            [a + "decoder_hidden_size"] = c.Int("decoder_hidden_size"),
            [a + "downsampling_ratios"] = Longs(c, "downsampling_ratios"),
            [a + "upsampling_ratios"] = Longs(c, "upsampling_ratios"),
            [a + "codebook_size"] = c.Int("codebook_size"),
            [a + "codebook_dim"] = c.Int("codebook_dim"),
            [a + "codebook_count"] = c.Int("n_codebooks"),
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            collector.Add(name.EndsWith(".alpha", StringComparison.Ordinal) ? HfTransforms.ToF32(info, name, new Shape(info.Shape[1].Fixed)) : HfTransforms.Plain(info, name));
        }

        return collector.Weights;
    }

    private static long[] Longs(HfConfig c, string key) => [.. c.Get(key)!.Value.EnumerateArray().Select(v => v.GetInt64())];
}
