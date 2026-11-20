using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.VoxCpm;

/// <summary>
/// VoxCPM importer: maps a VoxCPM2 checkpoint directory (OpenBMB's release: <c>config.json</c> with <c>"architecture": "voxcpm2"</c>, <c>model.safetensors</c>, the AudioVAE V2 as
/// <c>audiovae.pth</c> or <c>audiovae.safetensors</c>, <c>tokenizer.json</c>) onto the canonical <see cref="ModelSource"/> of architecture <c>voxcpm2</c>.
/// </summary>
/// <remarks>
/// <para>The text-semantic language model (<c>base_lm</c>, MiniCPM-4 without muP multipliers, LongRoPE) is mapped like a Hugging Face Llama checkpoint to the canonical decoder names and
/// <c>decoder.*</c> keys (unprefixed, as the decoder adapter builds them); the residual acoustic language model (<c>residual_lm</c>, no rotary embedding) the same way with its keys under
/// <c>voxcpm2.residual.</c> and its weights under <c>residual.</c>. Neither has an output vocabulary of its own (the host reads their hidden states): the stop head's last layer
/// (<c>stop_head.weight</c> [2, hidden]) stands in as the output head of both and as the residual model's embedding table. The local feature encoder (<c>feat_encoder.</c>), the local
/// diffusion transformer (<c>feat_decoder.estimator.</c>), the scalar quantizer (<c>fsq_layer.</c>), the projections and the stop predictor keep their checkpoint names; the AudioVAE's
/// tensors go under <c>audio_vae.</c> with the weight normalizations folded and the snake parameters as <c>[C]</c>. The hyper-parameters are documented on the adapter
/// (<c>VoxCpm2Adapter</c>).</para>
/// <para>Register with <c>services.AddVoxCpmImporter()</c> so that the <see cref="ModelLoader"/> opens VoxCPM2 checkpoints, or call <see cref="Open"/>.</para>
/// </remarks>
public sealed class VoxCpmImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "voxcpm2";

    private const string A = Architecture + ".";
    private const string Gain = ".weight_g";
    private const string Direction = ".weight_v";

    // The special tokens of the release (VoxCPM2Model: audio_start_token ... ref_audio_end_token).
    private const int AudioStart = 101;
    private const int AudioEnd = 102;
    private const int ReferenceStart = 103;
    private const int ReferenceEnd = 104;

    /// <inheritdoc/>
    public string Format => "voxcpm";

    /// <inheritdoc/>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path)
    {
        if (!Directory.Exists(path) || !File.Exists(Path.Combine(path, "model.safetensors")) || !File.Exists(Path.Combine(path, "config.json"))
            || !(File.Exists(Path.Combine(path, "audiovae.pth")) || File.Exists(Path.Combine(path, "audiovae.safetensors"))))
        {
            return false;
        }

        try
        {
            using var config = JsonDocument.Parse(File.ReadAllText(Path.Combine(path, "config.json")));
            return config.RootElement.TryGetProperty("architecture", out var architecture) && architecture.GetString() == Architecture;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens a VoxCPM2 checkpoint directory.</summary>
    /// <param name="path">The directory.</param>
    /// <exception cref="InvalidDataException">The directory is not a VoxCPM2 checkpoint, or a tensor is missing.</exception>
    public static ModelSource Open(string path)
    {
        var owned = new List<IDisposable>();
        try
        {
            using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(path, "config.json")));
            var root = document.RootElement;
            if (!root.TryGetProperty("architecture", out var architecture) || architecture.GetString() != Architecture)
            {
                throw new InvalidDataException($"{path}: not a VoxCPM2 checkpoint (config.json architecture '{architecture}').");
            }

            var model = SafetensorsFile.Open(Path.Combine(path, "model.safetensors"));
            owned.Add(model);
            string vaeSafetensors = Path.Combine(path, "audiovae.safetensors");
            var vae = File.Exists(vaeSafetensors) ? SafetensorsFile.Open(vaeSafetensors) : SafetensorsFile.OpenTorch(Path.Combine(path, "audiovae.pth"));
            owned.Add(vae);

            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
            var lm = root.GetProperty("lm_config");
            int maxLength = Int(root, "max_length", 8192);
            var stopHead = Tensor(model, "stop_head.weight");
            MapLanguageModels(model, root, lm, stopHead, maxLength, weights, metadata);

            string[] own = ["feat_encoder.", "feat_decoder.estimator.", "fsq_layer.", "enc_to_lm_proj.", "lm_to_dit_proj.", "res_to_dit_proj.", "fusion_concat_proj.", "stop_proj.", "stop_head."];
            foreach (var (name, tensor) in model.Tensors)
            {
                if (name == "feat_encoder.special_token")
                {
                    var w = tensor.ToWeightInfo();
                    weights[name] = w.As(name, new Shape(1, w.Shape.Elements()));
                }
                else if (own.Any(p => name.StartsWith(p, StringComparison.Ordinal)))
                {
                    weights[name] = HfTransforms.Plain(tensor.ToWeightInfo(), name);
                }
                else if (!name.StartsWith("base_lm.", StringComparison.Ordinal) && !name.StartsWith("residual_lm.", StringComparison.Ordinal))
                {
                    throw new InvalidDataException($"{path}: unexpected VoxCPM2 tensor '{name}'.");
                }
            }

            MapAudioVae(vae, weights);
            using (var tokenizerConfig = HfConfig.Parse("{}", path))
            {
                HfTokenizerTranslator.AddTo(metadata, path, tokenizerConfig);
            }

            metadata[CanonicalKeys.Architecture] = Architecture;
            metadata[CanonicalKeys.Name] = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar));
            AddHyperParameters(root, lm, maxLength, metadata);
            return new ModelSource
            {
                Format = "voxcpm",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = weights,
                OwnedResources = owned,
                Path = path,
            };
        }
        catch
        {
            foreach (var resource in owned)
            {
                resource.Dispose();
            }

            throw;
        }
    }

    // base_lm through the Hugging Face decoder mapping as a Llama; residual_lm likewise under residual. (its keys under voxcpm2.residual.).
    private static void MapLanguageModels(SafetensorsFile file, JsonElement root, JsonElement lm, SafetensorsTensor stopHead, int maxLength,
        Dictionary<string, WeightInfo> weights, Dictionary<string, object> metadata)
    {
        var mapping = new DecoderHfMapping();
        var options = new HuggingFaceImportOptions();
        var baseTensors = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal) { ["lm_head.weight"] = stopHead };
        var residualTensors = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal) { ["lm_head.weight"] = stopHead, ["model.embed_tokens.weight"] = stopHead };
        foreach (var (name, tensor) in file.Tensors)
        {
            if (name.StartsWith("base_lm.", StringComparison.Ordinal))
            {
                baseTensors["model." + name["base_lm.".Length..]] = tensor;
            }
            else if (name.StartsWith("residual_lm.", StringComparison.Ordinal))
            {
                residualTensors["model." + name["residual_lm.".Length..]] = tensor;
            }
        }

        using (var config = LlamaConfig(lm, lm.GetProperty("num_hidden_layers").GetInt32(), lm.GetProperty("vocab_size").GetInt32(), rope: true))
        {
            foreach (var (key, value) in mapping.MapHyperParameters(config, options))
            {
                metadata[key] = value;
            }

            foreach (var (name, w) in mapping.MapWeights(config, baseTensors, options))
            {
                weights[name] = w;
            }
        }

        metadata["decoder.context_length"] = maxLength;
        int residualLayers = Int(root, "residual_lm_num_layers", 8);
        bool noRope = root.TryGetProperty("residual_lm_no_rope", out var flag) && flag.GetBoolean();
        using (var config = LlamaConfig(lm, residualLayers, (int)stopHead.Shape[0], rope: !noRope))
        {
            foreach (var (key, value) in mapping.MapHyperParameters(config, options))
            {
                metadata[A + "residual." + key] = value;
            }

            foreach (var (name, w) in mapping.MapWeights(config, residualTensors, options))
            {
                weights["residual." + name] = w.As("residual." + name);
            }
        }

        metadata[A + "residual.decoder.context_length"] = maxLength;
        if (noRope)
        {
            metadata[A + "residual.decoder.rope.layers"] = new int[residualLayers];
        }
    }

    // A Llama configuration of the MiniCPM-4 language model (use_mup false: no multipliers), with the release's LongRoPE factors when the model rotates.
    private static HfConfig LlamaConfig(JsonElement lm, int layers, int vocabulary, bool rope)
    {
        int hidden = lm.GetProperty("hidden_size").GetInt32();
        int heads = lm.GetProperty("num_attention_heads").GetInt32();
        if (lm.TryGetProperty("use_mup", out var mup) && mup.GetBoolean())
        {
            throw new NotSupportedException("voxcpm2: language models with muP multipliers (use_mup) are not supported.");
        }

        var node = new JsonObject
        {
            ["model_type"] = "llama",
            ["architectures"] = new JsonArray("LlamaForCausalLM"),
            ["vocab_size"] = vocabulary,
            ["hidden_size"] = hidden,
            ["intermediate_size"] = lm.GetProperty("intermediate_size").GetInt32(),
            ["num_hidden_layers"] = layers,
            ["num_attention_heads"] = heads,
            ["num_key_value_heads"] = lm.GetProperty("num_key_value_heads").GetInt32(),
            ["head_dim"] = lm.TryGetProperty("kv_channels", out var kv) && kv.ValueKind == JsonValueKind.Number ? kv.GetInt32() : hidden / heads,
            ["hidden_act"] = "silu",
            ["rms_norm_eps"] = lm.GetProperty("rms_norm_eps").GetDouble(),
            ["max_position_embeddings"] = lm.GetProperty("max_position_embeddings").GetInt32(),
            ["rope_theta"] = lm.GetProperty("rope_theta").GetDouble(),
            ["tie_word_embeddings"] = false,
        };
        if (rope && lm.TryGetProperty("rope_scaling", out var scaling) && scaling.ValueKind == JsonValueKind.Object)
        {
            var copy = JsonNode.Parse(scaling.GetRawText())!.AsObject();
            copy["rope_type"] = copy["type"]?.GetValue<string>() ?? "longrope";
            node["rope_scaling"] = copy;
        }

        return HfConfig.Parse(node.ToJsonString(), ".");
    }

    // The AudioVAE V2: weight normalizations folded, snake parameters [1, C, 1] as [C]; the encoder's log-variance head (unused for inference) and the sample-rate
    // bucket boundaries (a hyper-parameter) are left out.
    private static void MapAudioVae(SafetensorsFile file, Dictionary<string, WeightInfo> weights)
    {
        const string Prefix = "audio_vae.";
        foreach (var (name, tensor) in file.Tensors)
        {
            if (name.EndsWith(Direction, StringComparison.Ordinal) || name.StartsWith("encoder.fc_logvar.", StringComparison.Ordinal) || name == "decoder.sr_bin_boundaries")
            {
                continue;
            }

            if (name.EndsWith(Gain, StringComparison.Ordinal))
            {
                string module = name[..^Gain.Length];
                string canonical = Prefix + module + ".weight";
                var direction = Tensor(file, module + Direction).ToWeightInfo();
                weights[canonical] = HfWeightNormFolding.Fold(tensor.ToWeightInfo(), direction, canonical);
                continue;
            }

            var w = tensor.ToWeightInfo();
            weights[Prefix + name] = name.EndsWith(".alpha", StringComparison.Ordinal)
                ? HfTransforms.Plain(w, Prefix + name).As(Prefix + name, new Shape(w.Shape.Elements()))
                : HfTransforms.Plain(w, Prefix + name);
        }
    }

    private static void AddHyperParameters(JsonElement root, JsonElement lm, int maxLength, Dictionary<string, object> md)
    {
        md[A + "patch_size"] = Int(root, "patch_size", 4);
        md[A + "feat_dim"] = Int(root, "feat_dim", 64);
        md[A + "fsq.latent"] = Int(root, "scalar_quantization_latent_dim", 512);
        md[A + "fsq.scale"] = Int(root, "scalar_quantization_scale", 9);
        md[A + "max_length"] = maxLength;
        md[A + "norm_eps"] = (float)lm.GetProperty("rms_norm_eps").GetDouble();
        md[A + "rope.base"] = (float)lm.GetProperty("rope_theta").GetDouble();
        md[A + "token.audio_start"] = AudioStart;
        md[A + "token.audio_end"] = AudioEnd;
        md[A + "token.reference_start"] = ReferenceStart;
        md[A + "token.reference_end"] = ReferenceEnd;
        int kvHeads = lm.GetProperty("num_key_value_heads").GetInt32();
        foreach (var (section, key) in new[] { ("encoder_config", "encoder"), ("dit_config", "dit") })
        {
            var s = root.GetProperty(section);
            int hidden = s.GetProperty("hidden_dim").GetInt32();
            int heads = s.GetProperty("num_heads").GetInt32();
            md[A + key + ".hidden_size"] = hidden;
            md[A + key + ".ffn_hidden_size"] = s.GetProperty("ffn_dim").GetInt32();
            md[A + key + ".head_count"] = heads;
            md[A + key + ".kv_head_count"] = kvHeads;
            md[A + key + ".head_dim"] = s.TryGetProperty("kv_channels", out var kv) && kv.ValueKind == JsonValueKind.Number ? kv.GetInt32() : hidden / heads;
            md[A + key + ".layer_count"] = s.GetProperty("num_layers").GetInt32();
        }

        var dit = root.GetProperty("dit_config");
        bool meanMode = (dit.TryGetProperty("dit_mean_mode", out var mean) || dit.TryGetProperty("mean_mode", out mean)) && mean.GetBoolean();
        md[A + "dit.mean_mode"] = meanMode;
        var cfm = dit.GetProperty("cfm_config");
        md[A + "sampling.guidance"] = cfm.TryGetProperty("inference_cfg_rate", out var cfg) ? (float)cfg.GetDouble() : 2f;
        md[A + "sampling.steps"] = 10;

        var vae = root.GetProperty("audio_vae_config");
        long[] Longs(string key) => [.. vae.GetProperty(key).EnumerateArray().Select(e => e.GetInt64())];
        int outRate = vae.TryGetProperty("out_sample_rate", out var o) ? o.GetInt32() : vae.GetProperty("sample_rate").GetInt32();
        md[A + "vae.encoder_dim"] = vae.GetProperty("encoder_dim").GetInt32();
        md[A + "vae.encoder_rates"] = Longs("encoder_rates");
        md[A + "vae.latent"] = vae.GetProperty("latent_dim").GetInt32();
        md[A + "vae.decoder_dim"] = vae.GetProperty("decoder_dim").GetInt32();
        md[A + "vae.decoder_rates"] = Longs("decoder_rates");
        md[A + "vae.sample_rate"] = vae.GetProperty("sample_rate").GetInt32();
        md[A + "vae.out_sample_rate"] = outRate;
        if (vae.TryGetProperty("depthwise", out var depthwise) && !depthwise.GetBoolean())
        {
            throw new NotSupportedException("voxcpm2: only the depthwise AudioVAE V2 is supported.");
        }

        if (vae.TryGetProperty("sr_bin_boundaries", out var boundaries) && boundaries.ValueKind == JsonValueKind.Array)
        {
            // torch.bucketize(out_sample_rate, boundaries): the number of boundaries below the rate.
            md[A + "vae.sample_rate_bucket"] = boundaries.EnumerateArray().Count(b => b.GetInt64() < outRate);
        }
    }

    private static int Int(JsonElement e, string key, int fallback) => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

    private static SafetensorsTensor Tensor(SafetensorsFile file, string name)
        => file.Tensors.TryGetValue(name, out var tensor) ? tensor : throw new InvalidDataException($"VoxCPM2: the tensor '{name}' is missing.");
}
