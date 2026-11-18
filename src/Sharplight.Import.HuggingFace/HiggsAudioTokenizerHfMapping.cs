using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// The Higgs Audio v2 tokenizer (<c>higgs_audio_v2_tokenizer</c>: a DAC-style acoustic encoder and decoder, a HuBERT semantic encoder with a convolutional adapter and a residual
/// vector quantizer with Euclidean codebooks; 24 kHz audio at 25 frames per second; the <c>audio_tokenizer</c> folder of OmniVoice) → canonical <c>higgs_audio_codec</c>
/// (documented on <c>HiggsAudioCodecAdapter</c>). The HuBERT maps through <see cref="Wav2Vec2HfMapping"/> under <c>semantic.</c> (its weight-normalized positional convolution
/// folded); the snake parameters are flattened to <c>[C]</c>; the semantic decoder, the semantic reconstruction head and the codebooks' EMA statistics (training only) are dropped.
/// </summary>
internal sealed partial class HiggsAudioTokenizerHfMapping : IHfArchitectureMapping
{
    private const string SemanticModel = "semantic_model.";

    /// <inheritdoc/>
    public string Architecture => "higgs_audio_codec";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["higgs_audio_v2_tokenizer"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var acoustic = c.Root.GetProperty("acoustic_model_config");
        var semantic = c.Root.GetProperty("semantic_model_config");
        string a = Architecture + ".";
        var down = acoustic.GetProperty("downsampling_ratios").EnumerateArray().Select(v => v.GetInt64()).ToArray();
        int sampleRate = c.Int("sample_rate");
        long hop = down.Aggregate(1L, (x, y) => x * y);
        int frameRate = (int)Math.Ceiling((double)sampleRate / hop);
        int codebookSize = c.Int("codebook_size");
        int bits = (int)Math.Ceiling(Math.Log2(codebookSize));
        double top = c.Root.GetProperty("target_bandwidths").EnumerateArray().Max(v => v.GetDouble());
        int codebooks = (int)(1000 * top / (frameRate * bits));
        int semanticRate = c.Int("semantic_sample_rate");
        int downsampleFactor = c.Int("downsample_factor");
        int semanticHidden = semantic.GetProperty("hidden_size").GetInt32();
        var strides = c.Root.GetProperty("strides").EnumerateArray().Select(v => v.GetInt64()).ToArray();
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "sample_rate"] = sampleRate,
            [a + "semantic_sample_rate"] = semanticRate,
            [a + "hidden_size"] = acoustic.GetProperty("hidden_size").GetInt32() + semanticHidden,
            [a + "codebook_size"] = codebookSize,
            [a + "codebook_dim"] = c.Int("codebook_dim"),
            [a + "codebook_count"] = codebooks,
            [a + "acoustic.hidden_size"] = acoustic.GetProperty("hidden_size").GetInt32(),
            [a + "acoustic.encoder_hidden_size"] = acoustic.GetProperty("encoder_hidden_size").GetInt32(),
            [a + "acoustic.decoder_hidden_size"] = acoustic.GetProperty("decoder_hidden_size").GetInt32(),
            [a + "acoustic.downsampling_ratios"] = down,
            [a + "acoustic.upsampling_ratios"] = acoustic.GetProperty("upsampling_ratios").EnumerateArray().Select(v => v.GetInt64()).ToArray(),
            [a + "semantic.hidden_size"] = semanticHidden,
            [a + "semantic.downsample"] = (int)(hop / ((double)sampleRate / semanticRate) / downsampleFactor),
            [a + "semantic.kernel_size"] = c.Int("kernel_size"),
            [a + "semantic.unit_kernel_size"] = c.Int("unit_kernel_size"),
            [a + "semantic.strides"] = strides,
            [a + "semantic.block_dilations"] = c.Root.GetProperty("block_dilations").EnumerateArray().Select(v => v.GetInt64()).ToArray(),
            [a + "hubert.hidden_size"] = semanticHidden,
            [a + "hubert.layers"] = semantic.GetProperty("num_hidden_layers").GetInt32(),
            [a + "hubert.heads"] = semantic.GetProperty("num_attention_heads").GetInt32(),
            [a + "hubert.conv.strides"] = semantic.GetProperty("conv_stride").EnumerateArray().Select(v => v.GetInt64()).ToArray(),
            [a + "hubert.position_conv.groups"] = semantic.GetProperty("num_conv_pos_embedding_groups").GetInt32(),
            [a + "hubert.norm_eps"] = (float)semantic.GetProperty("layer_norm_eps").GetDouble(),
        };
    }

    [GeneratedRegex(@"^quantizer\.quantizers\.\d+\.codebook\.(cluster_size|embed_avg|inited)$")]
    private static partial Regex TrainingOnly();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        var hubert = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (name.StartsWith(SemanticModel, StringComparison.Ordinal))
            {
                hubert[name[SemanticModel.Length..]] = tensor;
            }
            else if (name.StartsWith("decoder_semantic.", StringComparison.Ordinal) || name.StartsWith("fc1.", StringComparison.Ordinal) || TrainingOnly().IsMatch(name))
            {
                continue;
            }
            else
            {
                var info = tensor.ToWeightInfo();
                collector.Add(name.EndsWith(".alpha", StringComparison.Ordinal) ? HfTransforms.ToF32(info, name, new Shape(info.Shape[1].Fixed)) : HfTransforms.Plain(info, name));
            }
        }

        Wav2Vec2HfMapping.MapInto(collector, hubert, "semantic.", ctcHead: false);
        return collector.Weights;
    }
}
