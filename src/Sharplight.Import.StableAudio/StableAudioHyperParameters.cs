using System.Text.Json;
using Sharplight.Import.HuggingFace;

namespace Sharplight.Import.StableAudio;

/// <summary>Translates a Stable Audio 3 <c>model_config.json</c> and the T5Gemma <c>config.json</c> into the canonical <c>stable_audio_3.*</c> keys.</summary>
internal static class StableAudioHyperParameters
{
    private const string A = StableAudioImporter.Architecture + ".";

    // StableAudioModel.generate decodes in chunks of 128 latent frames with 32 frames of overlap (AudioAutoencoder.decode_audio's defaults).
    private const int DecodeFrames = 128;
    private const int DecodeOverlap = 32;

    // SoftNormBottleneck.decode adds noise of 1e-3 times the latent scale in evaluation mode when noise_regularize is set.
    private const float LatentNoise = 1e-3f;

    /// <summary>Adds the canonical hyper-parameters.</summary>
    /// <param name="root">The parsed <c>model_config.json</c>.</param>
    /// <param name="text">The T5Gemma folder's <c>config.json</c>.</param>
    /// <param name="differential">Whether the diffusion transformer uses differential attention.</param>
    /// <param name="md">The canonical metadata being built.</param>
    /// <exception cref="NotSupportedException">The configuration uses an unsupported option.</exception>
    public static void Add(JsonElement root, HfConfig text, bool differential, Dictionary<string, object> md)
    {
        var model = root.GetProperty("model");
        var diffusion = model.GetProperty("diffusion");
        var dit = diffusion.GetProperty("config");
        var pretransform = model.GetProperty("pretransform").GetProperty("config");
        var encoder = pretransform.GetProperty("encoder").GetProperty("config");
        var decoder = pretransform.GetProperty("decoder").GetProperty("config");
        var conditioning = model.GetProperty("conditioning");

        string objective = diffusion.GetProperty("diffusion_objective").GetString()!;
        if (objective is not ("rectified_flow" or "rf_denoiser"))
        {
            throw new NotSupportedException($"Stable Audio 3: the diffusion objective '{objective}' is not supported.");
        }

        if (Int(dit, "patch_size", 1) != 1 || Str(dit, "global_cond_type", "prepend") != "adaLN" || Str(dit, "timestep_features_type", "learned") != "expo"
            || Str(dit, "norm_type", "layer_norm") != "rms_norm" || Str(dit.GetProperty("attn_kwargs"), "qk_norm", "none") != "rms")
        {
            throw new NotSupportedException("Stable Audio 3: only the adaLN diffusion transformer with RMS norms and exponential Fourier time features is supported.");
        }

        int sampleRate = root.GetProperty("sample_rate").GetInt32();
        int ratio = pretransform.GetProperty("downsampling_ratio").GetInt32();
        int width = dit.GetProperty("embed_dim").GetInt32();
        int heads = dit.GetProperty("num_heads").GetInt32();
        md[A + "objective"] = objective;
        md[A + "sample_rate"] = sampleRate;
        md[A + "audio_channels"] = root.GetProperty("audio_channels").GetInt32();
        md[A + "latent_channels"] = model.GetProperty("io_channels").GetInt32();
        md[A + "samples_per_frame"] = ratio;
        md[A + "max_frames"] = (int)(root.GetProperty("sample_size").GetInt64() / ratio);
        md[A + "frame_align"] = Int(encoder, "chunk_size", 32) / encoder.GetProperty("strides")[0].GetInt32();
        md[A + "headroom_seconds"] = 6f;
        md[A + "hidden_size"] = width;
        md[A + "layer_count"] = dit.GetProperty("depth").GetInt32();
        md[A + "head_count"] = heads;
        md[A + "differential"] = differential;
        md[A + "memory_tokens"] = Int(dit, "num_memory_tokens", 0);
        md[A + "rope_dims"] = Math.Max(width / heads / 2, 32);
        md[A + "time_features"] = Int(dit, "timestep_features_dim", 256);
        md[A + "time_logsnr"] = dit.TryGetProperty("timestep_features_logsnr", out var logsnr) && logsnr.GetBoolean();
        md[A + "condition_width"] = conditioning.GetProperty("cond_dim").GetInt32();
        md[A + "local_channels"] = Int(dit, "local_add_cond_dim", 0);
        md[A + "default_steps"] = objective == "rf_denoiser" ? 8 : 50;
        md[A + "default_guidance"] = objective == "rf_denoiser" ? 1f : 7f;
        AddConditioners(conditioning, md);
        AddSchedule(diffusion, md);
        AddDecoder(pretransform, decoder, md);
        AddTextEncoder(text, md);
    }

    private static void AddConditioners(JsonElement conditioning, Dictionary<string, object> md)
    {
        foreach (var c in conditioning.GetProperty("configs").EnumerateArray())
        {
            var config = c.GetProperty("config");
            switch (c.GetProperty("type").GetString())
            {
                case "t5gemma":
                    if (Str(config, "padding_mode", "zero") != "learned")
                    {
                        throw new NotSupportedException("Stable Audio 3: only the learned padding of the prompt conditioner is supported.");
                    }

                    md[A + "prompt_length"] = Int(config, "max_length", 128);
                    break;
                case "number" when c.GetProperty("id").GetString() == "seconds_total":
                    md[A + "seconds.min"] = (float)Num(config, "min_val", 0);
                    md[A + "seconds.max"] = (float)Num(config, "max_val", 1);
                    break;
                default:
                    throw new NotSupportedException($"Stable Audio 3: the conditioner '{c.GetProperty("id").GetString()}' is not supported.");
            }
        }
    }

    // The sampling schedule: sampling_distribution_shift_options, by default the length-invariant log-SNR spacing of the package.
    private static void AddSchedule(JsonElement diffusion, Dictionary<string, object> md)
    {
        if (!diffusion.TryGetProperty("sampling_distribution_shift_options", out var options) || options.ValueKind != JsonValueKind.Object)
        {
            md[A + "schedule.type"] = "logsnr";
            md[A + "schedule.rate"] = 0f;
            md[A + "schedule.anchor_logsnr"] = -6.2f;
            md[A + "schedule.anchor_length"] = 2000f;
            md[A + "schedule.logsnr_end"] = 2f;
            return;
        }

        string type = Str(options, "type", "full");
        md[A + "schedule.type"] = type;
        switch (type)
        {
            case "none":
                break;
            case "logsnr":
                md[A + "schedule.rate"] = (float)Num(options, "rate", 1);
                md[A + "schedule.anchor_logsnr"] = (float)Num(options, "anchor_logsnr", -6.2);
                md[A + "schedule.anchor_length"] = (float)Num(options, "anchor_length", 2000);
                md[A + "schedule.logsnr_end"] = (float)Num(options, "logsnr_end", 2);
                break;
            case "flux":
                md[A + "schedule.min_length"] = (float)Num(options, "min_length", 256);
                md[A + "schedule.max_length"] = (float)Num(options, "max_length", 4096);
                md[A + "schedule.alpha_min"] = (float)Num(options, "alpha_min", 1);
                md[A + "schedule.alpha_max"] = (float)Num(options, "alpha_max", 1);
                break;
            case "full":
                if (options.TryGetProperty("use_sine", out var sine) && sine.GetBoolean())
                {
                    throw new NotSupportedException("Stable Audio 3: the sine variant of the sampling distribution shift is not supported.");
                }

                md[A + "schedule.min_length"] = (float)Num(options, "min_length", 256);
                md[A + "schedule.max_length"] = (float)Num(options, "max_length", 4096);
                md[A + "schedule.base_shift"] = (float)Num(options, "base_shift", 0.5);
                md[A + "schedule.max_shift"] = (float)Num(options, "max_shift", 1.15);
                break;
            default:
                throw new NotSupportedException($"Stable Audio 3: the sampling distribution shift '{type}' is not supported.");
        }
    }

    private static void AddDecoder(JsonElement pretransform, JsonElement decoder, Dictionary<string, object> md)
    {
        var mults = decoder.GetProperty("c_mults");
        var strides = decoder.GetProperty("strides");
        if (mults.GetArrayLength() != 1 || pretransform.GetProperty("pretransform").GetProperty("type").GetString() != "patched")
        {
            throw new NotSupportedException("Stable Audio 3: only the single-stage SAME decoder over a patched waveform is supported.");
        }

        var patched = pretransform.GetProperty("pretransform").GetProperty("config");
        if (Int(patched, "oversampling", 1) != 1 || Int(patched, "postfilter_channels", 0) != 0 || Bool(decoder, "layer_scale") || Bool(decoder, "conformer")
            || Bool(decoder, "feat_scale") || Bool(decoder, "cross_attn"))
        {
            throw new NotSupportedException("Stable Audio 3: this SAME decoder variant is not supported.");
        }

        var bottleneck = pretransform.GetProperty("bottleneck");
        md[A + "vae.width"] = decoder.GetProperty("channels").GetInt32() * mults[0].GetInt32();
        md[A + "vae.out_channels"] = decoder.GetProperty("out_channels").GetInt32();
        md[A + "vae.stride"] = strides[0].GetInt32();
        md[A + "vae.layer_count"] = decoder.GetProperty("transformer_depths")[0].GetInt32();
        md[A + "vae.head_dim"] = Int(decoder, "dim_heads", 128);
        md[A + "vae.differential"] = !decoder.TryGetProperty("differential", out var differential) || differential.GetBoolean();
        md[A + "vae.dyt"] = !decoder.TryGetProperty("dyt", out var dyt) || dyt.GetBoolean();
        md[A + "vae.chunk_size"] = Int(decoder, "chunk_size", 128);
        md[A + "vae.midpoint_shift"] = Bool(decoder, "chunk_midpoint_shift");
        md[A + "vae.sliding_window"] = decoder.TryGetProperty("sliding_window", out var window) && window.ValueKind == JsonValueKind.Array
            ? window.EnumerateArray().Select(w => w.GetInt64()).ToArray()
            : Array.Empty<long>();
        md[A + "vae.sinusoidal_blocks"] = decoder.TryGetProperty("sinusoidal_blocks", out var sinusoidal) ? sinusoidal[0].GetInt32() : 0;
        md[A + "vae.mapping_kernel"] = Bool(decoder, "conv_mapping") ? 3 : 1;
        md[A + "vae.patch_size"] = patched.GetProperty("patch_size").GetInt32();
        md[A + "vae.mask_noise"] = (float)Num(decoder, "mask_noise", 0);
        md[A + "vae.latent_noise"] = Bool(bottleneck.GetProperty("config"), "noise_regularize") ? LatentNoise : 0f;
        md[A + "vae.soft_clip"] = Bool(pretransform.GetProperty("decoder"), "soft_clip");
        md[A + "vae.decode_frames"] = DecodeFrames;
        md[A + "vae.decode_overlap"] = DecodeOverlap;
    }

    private static void AddTextEncoder(HfConfig text, Dictionary<string, object> md)
    {
        var e = text.Root.GetProperty("encoder");
        if (Str(e, "hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh")
        {
            throw new NotSupportedException("Stable Audio 3: only the tanh-approximated GELU of T5Gemma is supported.");
        }

        md[A + "text.hidden_size"] = e.GetProperty("hidden_size").GetInt32();
        md[A + "text.layer_count"] = e.GetProperty("num_hidden_layers").GetInt32();
        md[A + "text.intermediate_size"] = e.GetProperty("intermediate_size").GetInt32();
        md[A + "text.head_count"] = e.GetProperty("num_attention_heads").GetInt32();
        md[A + "text.kv_head_count"] = e.GetProperty("num_key_value_heads").GetInt32();
        md[A + "text.head_dim"] = e.GetProperty("head_dim").GetInt32();
        md[A + "text.norm_eps"] = (float)Num(e, "rms_norm_eps", 1e-6);
        md[A + "text.rope_base"] = (float)Num(e, "rope_theta", 10000);
        md[A + "text.softcap"] = (float)Num(e, "attn_logit_softcapping", 0);
        md[A + "text.query_scale"] = (float)Num(e, "query_pre_attn_scalar", e.GetProperty("head_dim").GetInt32());
        md[A + "text.sliding_window"] = Int(e, "sliding_window", 0);
        md[A + "text.sliding_layers"] = e.TryGetProperty("layer_types", out var types)
            ? types.EnumerateArray().Select(t => t.GetString() == "sliding_attention" ? 1L : 0L).ToArray()
            : Array.Empty<long>();
    }

    private static int Int(JsonElement e, string name, int fallback) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

    private static double Num(JsonElement e, string name, double fallback) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : fallback;

    private static string Str(JsonElement e, string name, string fallback) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString()! : fallback;

    private static bool Bool(JsonElement e, string name) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.True;
}
