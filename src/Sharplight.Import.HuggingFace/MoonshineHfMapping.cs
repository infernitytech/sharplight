using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face Moonshine checkpoints (<c>moonshine</c>: UsefulSensors/moonshine-tiny, -base) → canonical <c>encoder_decoder</c> with a
/// variable-length audio encoder over the raw waveform (documented on <c>EncoderDecoderAdapter</c>, <c>AudioConvStem</c> and
/// <c>SpeechRecognitionDescription</c>): three unpadded convolutions (tanh, a one-group norm, GELU, GELU), pre-norm LayerNorm layers without
/// biases, rotary attention on the first <c>partial_rotary_factor</c> of every head (GPT-J pairs) in the encoder and in the decoder's
/// self-attention, a GELU encoder MLP and a SwiGLU decoder MLP (the fused <c>fc1</c> split into its value and gate halves), and a decoder
/// prompted with the start token only (no language, task or timestamp tokens). The checkpoint's head-dimension padding
/// (<c>pad_head_dim_to_multiple_of</c>) only aligns the reference attention kernels and does not change the result, so it is not mapped.
/// </summary>
internal sealed partial class MoonshineHfMapping : IHfArchitectureMapping
{
    /// <summary>The longest window of the encoder, in seconds (the decoder's 194-token context bounds the text of one window).</summary>
    private const int WindowSeconds = 30;

    private static readonly Dictionary<string, string> s_layerModules = new(StringComparer.Ordinal)
    {
        ["self_attn.q_proj"] = "attention.q",
        ["self_attn.k_proj"] = "attention.k",
        ["self_attn.v_proj"] = "attention.v",
        ["self_attn.o_proj"] = "attention.o",
        ["encoder_attn.q_proj"] = "cross_attention.q",
        ["encoder_attn.k_proj"] = "cross_attention.k",
        ["encoder_attn.v_proj"] = "cross_attention.v",
        ["encoder_attn.o_proj"] = "cross_attention.o",
        ["mlp.fc2"] = "mlp.down",
    };

    /// <inheritdoc/>
    public string Architecture => "encoder_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["moonshine"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig config)
    {
        int heads = config.GetInt("encoder_num_attention_heads") ?? 8;
        if (config.GetInt("decoder_num_attention_heads") is { } decoderHeads && decoderHeads != heads)
        {
            yield return "moonshine: different encoder and decoder head counts are not supported.";
        }

        if ((config.GetInt("encoder_num_key_value_heads") ?? heads) != heads || (config.GetInt("decoder_num_key_value_heads") ?? heads) != heads)
        {
            yield return "moonshine: grouped key-value heads are not supported.";
        }

        if (config.GetBool("attention_bias") == true)
        {
            yield return "moonshine: attention biases are not supported.";
        }

        if (config.Get("rope_scaling") is { ValueKind: System.Text.Json.JsonValueKind.Object })
        {
            yield return "moonshine: rope scaling is not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        int hidden = c.Int("hidden_size");
        int heads = c.GetInt("encoder_num_attention_heads") ?? c.Int("num_attention_heads");
        int headDim = hidden / heads;
        double partial = c.GetDouble("partial_rotary_factor") ?? 0.9;
        int sampleRate = 16000;
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "encoder.layer_count"] = c.GetInt("encoder_num_hidden_layers") ?? c.Int("num_hidden_layers"),
            [a + "decoder.layer_count"] = c.GetInt("decoder_num_hidden_layers") ?? c.Int("num_hidden_layers"),
            [a + "attention.head_count"] = heads,
            [a + "attention.head_dim"] = headDim,
            [a + "attention.scale"] = 1f / MathF.Sqrt(headDim),
            [a + "attention.rope.dims"] = (long)(int)(headDim * partial),
            [a + "attention.rope.base"] = (float)(c.GetDouble("rope_theta") ?? 10000.0),
            [a + "attention.rope.style"] = "gptj",
            [a + "norm"] = "layer",
            [a + "norm_eps"] = 1e-5f,
            [a + "pre_norm"] = true,
            [a + "encoder.mlp.activation"] = c.GetString("encoder_hidden_act") ?? "gelu",
            [a + "decoder.mlp.activation"] = c.GetString("decoder_hidden_act") ?? "silu",
            [a + "decoder_start_token"] = c.GetInt("decoder_start_token_id") ?? 1,
            [a + "context_length"] = c.GetInt("max_position_embeddings") ?? 194,
            [a + "encoder.input"] = "audio",
            [a + "encoder.conv.strides"] = new long[] { 64, 3, 2 },
            [a + "encoder.conv.padding"] = new long[] { 0, 0, 0 },
            [a + "encoder.conv.activations"] = new[] { "tanh", "gelu", "gelu" },
            [a + "cross_attention.cached"] = true,
            [a + "audio.features"] = "waveform",
            [a + "audio.normalize"] = ReadNormalize(c),
            [a + "audio.channels"] = 1,
            [a + "audio.variable_length"] = true,
            [a + "audio.max_frames"] = (long)WindowSeconds * sampleRate,
            [a + "audio.sample_rate"] = sampleRate,
        };
    }

    [GeneratedRegex(@"^(encoder|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex LayerName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        int intermediate = c.Int("intermediate_size");
        foreach (var (rawName, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            string name = rawName.StartsWith("model.", StringComparison.Ordinal) ? rawName["model.".Length..] : rawName;
            switch (name)
            {
                case "decoder.embed_tokens.weight":
                    collector.Add(info.As("token_embedding.weight"));
                    continue;
                case "proj_out.weight":
                    collector.Add(info.As("output.weight"));
                    continue;
                case "encoder.layer_norm.weight" or "decoder.norm.weight":
                    collector.Add(HfTransforms.Plain(info, name[..name.IndexOf('.', StringComparison.Ordinal)] + ".output_norm.weight"));
                    continue;
                case "encoder.conv1.weight" or "encoder.conv2.weight" or "encoder.conv3.weight":
                    collector.Add(info.As($"encoder.conv.{name["encoder.conv".Length] - '1'}.weight"));
                    continue;
                case "encoder.conv2.bias" or "encoder.conv3.bias":
                    collector.Add(HfTransforms.Plain(info, $"encoder.conv.{name["encoder.conv".Length] - '1'}.bias"));
                    continue;
                case "encoder.groupnorm.weight" or "encoder.groupnorm.bias":
                    collector.Add(HfTransforms.Plain(info, "encoder.conv.0.norm." + name[(name.LastIndexOf('.') + 1)..]));
                    continue;
            }

            var m = LayerName().Match(name);
            if (!m.Success)
            {
                throw new NotSupportedException($"moonshine: tensor '{rawName}' has no canonical mapping.");
            }

            string stack = m.Groups[1].Value;
            string prefix = $"{stack}.layers.{m.Groups[2].Value}.";
            string module = m.Groups[3].Value;
            string kind = m.Groups[4].Value;
            if (module == "mlp.fc1" && stack == "decoder")
            {
                // SwiGLU: the first half of fc1 is the value, the second half the gate (act(gate) · value).
                collector.Add(HfTransforms.Plain(info.RowSlice("", 0, intermediate), prefix + "mlp.up." + kind));
                collector.Add(HfTransforms.Plain(info.RowSlice("", intermediate, intermediate), prefix + "mlp.gate." + kind));
                continue;
            }

            string? canonical = module switch
            {
                "mlp.fc1" => "mlp.up",
                "input_layernorm" => "attention_norm",
                "post_attention_layernorm" => stack == "encoder" ? "mlp_norm" : "cross_attention_norm",
                "final_layernorm" => "mlp_norm",
                _ => s_layerModules.GetValueOrDefault(module),
            };
            if (canonical is null)
            {
                throw new NotSupportedException($"moonshine: tensor '{rawName}' has no canonical mapping.");
            }

            collector.Add(HfTransforms.Plain(info, prefix + canonical + "." + kind));
        }

        return collector.Weights;
    }

    /// <inheritdoc/>
    public void AddMetadata(IDictionary<string, object> metadata, HfConfig config, string directory)
    {
        metadata[SpeechKeys.StartId] = config.GetInt("decoder_start_token_id") ?? config.Int("bos_token_id");
        metadata[SpeechKeys.EndId] = config.Int("eos_token_id");
        metadata[SpeechKeys.MaxTextTokens] = (config.GetInt("max_position_embeddings") ?? 194) - 1;
    }

    // Wav2Vec2FeatureExtractor's do_normalize (Moonshine ships false).
    private static bool ReadNormalize(HfConfig config)
    {
        string path = Path.Combine(config.Directory, "preprocessor_config.json");
        if (!File.Exists(path))
        {
            return false;
        }

        using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(path));
        return document.RootElement.TryGetProperty("do_normalize", out var value) && value.ValueKind == System.Text.Json.JsonValueKind.True;
    }
}
