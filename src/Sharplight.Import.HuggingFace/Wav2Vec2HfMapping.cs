using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face wav2vec2 CTC checkpoints (<c>Wav2Vec2ForCTC</c>: facebook/wav2vec2-base-960h, the XLSR-53 fine-tunes of jonatasgrosman and
/// others) → canonical <c>wav2vec2_ctc</c> (documented on <c>Wav2Vec2CtcAdapter</c>): the convolutional feature encoder over the raw waveform
/// (a group norm after the first convolution, or a layer norm after every one), the feature projection, the grouped positional convolution
/// with its weight normalization folded (both the <c>weight_g</c> / <c>weight_v</c> and the parametrization names), a post-norm (base) or
/// pre-norm (<c>do_stable_layer_norm</c>, large and XLSR) transformer encoder, and the CTC head. The character vocabulary becomes a Unigram
/// tokenizer (see <c>HfTokenizerTranslator</c>), the word delimiter <c>|</c> the SentencePiece space.
/// </summary>
internal sealed partial class Wav2Vec2HfMapping : IHfArchitectureMapping
{
    private const int WindowSeconds = 30;

    private static readonly Dictionary<string, string> s_layerModules = new(StringComparer.Ordinal)
    {
        ["attention.q_proj"] = "attention.q",
        ["attention.k_proj"] = "attention.k",
        ["attention.v_proj"] = "attention.v",
        ["attention.out_proj"] = "attention.o",
        ["layer_norm"] = "attention_norm",
        ["feed_forward.intermediate_dense"] = "mlp.up",
        ["feed_forward.output_dense"] = "mlp.down",
        ["final_layer_norm"] = "mlp_norm",
    };

    /// <inheritdoc/>
    public string Architecture => "wav2vec2_ctc";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["wav2vec2"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (!c.Architectures.Any(a => a == "Wav2Vec2ForCTC"))
        {
            yield return "wav2vec2: only Wav2Vec2ForCTC checkpoints (with a CTC head) are supported.";
        }

        string norm = c.GetString("feat_extract_norm") ?? "group";
        if (norm is not ("group" or "layer"))
        {
            yield return $"wav2vec2: feat_extract_norm '{norm}' is not supported.";
        }

        if (c.GetBool("do_stable_layer_norm") != true && norm == "layer")
        {
            yield return "wav2vec2: layer-normed feature encoders with a post-norm transformer are not supported.";
        }

        if (c.Get("adapter_kernel_size") is not null && c.GetBool("add_adapter") == true)
        {
            yield return "wav2vec2: the adapter layers are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        int sampleRate = ReadPreprocessor(c, out bool normalize);
        var strides = ReadLongs(c, "conv_stride");
        string norm = c.GetString("feat_extract_norm") ?? "group";
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "encoder.layer_count"] = c.Int("num_hidden_layers"),
            [a + "encoder.attention.head_count"] = c.Int("num_attention_heads"),
            [a + "encoder.mlp.activation"] = c.GetString("hidden_act") ?? "gelu",
            [a + "encoder.norm_eps"] = (float)(c.GetDouble("layer_norm_eps") ?? 1e-5),
            [a + "encoder.stable_layer_norm"] = c.GetBool("do_stable_layer_norm") == true,
            [a + "encoder.conv.strides"] = strides,
            [a + "encoder.conv.padding"] = new long[strides.Length],
            [a + "encoder.conv.activation"] = c.GetString("feat_extract_activation") ?? "gelu",
            [a + "encoder.conv.norm"] = norm,
            [a + "encoder.conv.norm_eps"] = 1e-5f,
            [a + "encoder.position_conv.groups"] = c.GetInt("num_conv_pos_embedding_groups") ?? 16,
            [a + "audio.features"] = "waveform",
            [a + "audio.normalize"] = normalize,
            [a + "audio.channels"] = 1,
            [a + "audio.variable_length"] = true,
            [a + "audio.max_frames"] = (long)WindowSeconds * sampleRate,
            [a + "audio.sample_rate"] = sampleRate,
            [a + "decoder.type"] = "ctc",
            [a + "decoder.blank_id"] = c.GetInt("pad_token_id") ?? 0,
        };
    }

    [GeneratedRegex(@"^(?:wav2vec2\.)?feature_extractor\.conv_layers\.(\d+)\.(conv|layer_norm)\.(weight|bias)$")]
    private static partial Regex ConvName();

    [GeneratedRegex(@"^(?:wav2vec2\.)?encoder\.layers\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex LayerName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        MapInto(collector, tensors, "", ctcHead: true);
        return collector.Weights;
    }

    /// <summary>
    /// Maps the tensors of a wav2vec2 or HuBERT encoder (names relative to the model, an optional <c>wav2vec2.</c> prefix is accepted) to the canonical names under
    /// <paramref name="targetPrefix"/>; the masked-span embedding and the pre-training heads are dropped.
    /// </summary>
    /// <param name="collector">Receives the weights.</param>
    /// <param name="tensors">The encoder's tensors.</param>
    /// <param name="targetPrefix">Prepended to every canonical name (empty, or <c>semantic.</c> for the speech codec that embeds a HuBERT).</param>
    /// <param name="ctcHead">Whether <c>lm_head</c> maps to the CTC head (it is dropped otherwise).</param>
    internal static void MapInto(HfWeightCollector collector, IReadOnlyDictionary<string, SafetensorsTensor> tensors, string targetPrefix, bool ctcHead)
    {
        string t = targetPrefix;
        foreach (var (rawName, tensor) in tensors)
        {
            string name = rawName.StartsWith("wav2vec2.", StringComparison.Ordinal) ? rawName["wav2vec2.".Length..] : rawName;
            var info = tensor.ToWeightInfo();
            switch (name)
            {
                case "masked_spec_embed" or "quantizer.weight_proj.weight" or "quantizer.weight_proj.bias" or "quantizer.codevectors" or "project_hid.weight"
                    or "project_hid.bias" or "project_q.weight" or "project_q.bias":
                    continue;
                case "lm_head.weight" or "lm_head.bias":
                    if (ctcHead)
                    {
                        collector.Add(HfTransforms.Plain(info, t + "ctc." + Suffix(name)));
                    }

                    continue;
                case "feature_projection.layer_norm.weight" or "feature_projection.layer_norm.bias":
                    collector.Add(HfTransforms.Plain(info, t + "encoder.projection_norm." + Suffix(name)));
                    continue;
                case "feature_projection.projection.weight" or "feature_projection.projection.bias":
                    collector.Add(HfTransforms.Plain(info, t + "encoder.projection." + Suffix(name)));
                    continue;
                case "encoder.layer_norm.weight" or "encoder.layer_norm.bias":
                    collector.Add(HfTransforms.Plain(info, t + "encoder.norm." + Suffix(name)));
                    continue;
                case "encoder.pos_conv_embed.conv.bias":
                    collector.Add(HfTransforms.Plain(info, t + "encoder.position_conv.bias"));
                    continue;
                case "encoder.pos_conv_embed.conv.weight":
                    collector.Add(HfTransforms.ToF32(info, t + "encoder.position_conv.weight"));
                    continue;
                case "encoder.pos_conv_embed.conv.weight_g" or "encoder.pos_conv_embed.conv.parametrizations.weight.original0":
                    collector.Add(HfWeightNormFolding.FoldLastAxis(info, FindDirection(tensors, name), t + "encoder.position_conv.weight"));
                    continue;
                case "encoder.pos_conv_embed.conv.weight_v" or "encoder.pos_conv_embed.conv.parametrizations.weight.original1":
                    continue;
            }

            var conv = ConvName().Match(name);
            if (conv.Success)
            {
                string target = conv.Groups[2].Value == "conv" ? "" : "norm.";
                collector.Add(HfTransforms.Plain(info, $"{t}encoder.conv.{conv.Groups[1].Value}.{target}{conv.Groups[3].Value}"));
                continue;
            }

            var layer = LayerName().Match(name);
            if (layer.Success && s_layerModules.TryGetValue(layer.Groups[2].Value, out var module))
            {
                collector.Add(HfTransforms.Plain(info, $"{t}encoder.layers.{layer.Groups[1].Value}.{module}.{layer.Groups[3].Value}"));
                continue;
            }

            throw new NotSupportedException($"wav2vec2: tensor '{rawName}' has no canonical mapping.");
        }
    }

    private static WeightInfo FindDirection(IReadOnlyDictionary<string, SafetensorsTensor> tensors, string gainName)
    {
        string direction = gainName.EndsWith("weight_g", StringComparison.Ordinal)
            ? gainName[..^1] + "v"
            : gainName[..^"original0".Length] + "original1";
        string key = tensors.Keys.First(k => k.EndsWith(direction, StringComparison.Ordinal));
        return tensors[key].ToWeightInfo();
    }

    private static string Suffix(string name) => name[(name.LastIndexOf('.') + 1)..];

    private static long[] ReadLongs(HfConfig c, string key)
        => c.Root.GetProperty(key).EnumerateArray().Select(e => e.GetInt64()).ToArray();

    // The feature extractor of preprocessor_config.json: sampling rate and do_normalize (Wav2Vec2FeatureExtractor defaults).
    private static int ReadPreprocessor(HfConfig c, out bool normalize)
    {
        normalize = true;
        string path = Path.Combine(c.Directory, "preprocessor_config.json");
        if (!File.Exists(path))
        {
            return 16000;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = document.RootElement;
        if (root.TryGetProperty("do_normalize", out var flag))
        {
            normalize = flag.ValueKind == JsonValueKind.True;
        }

        return root.TryGetProperty("sampling_rate", out var rate) && rate.ValueKind == JsonValueKind.Number ? rate.GetInt32() : 16000;
    }
}
