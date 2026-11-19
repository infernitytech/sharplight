using System.Text.RegularExpressions;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.StableAudio;

/// <summary>Maps the tensors of a Stable Audio 3 checkpoint onto the canonical names of <c>StableAudio3Adapter</c>.</summary>
internal static partial class StableAudioWeights
{
    private const string Dit = "model.model.";
    private const string Decoder = "pretransform.model.decoder.layers.";

    /// <summary>Adds the conditioners, the diffusion transformer and the autoencoder's decoder of <c>model.safetensors</c>.</summary>
    /// <param name="tensors">The tensors of <c>model.safetensors</c>.</param>
    /// <param name="weights">Receives the canonical weights.</param>
    /// <exception cref="NotSupportedException">A tensor of the transformer or the decoder has no canonical name (an unsupported variant).</exception>
    public static void Map(IReadOnlyDictionary<string, SafetensorsTensor> tensors, Dictionary<string, WeightInfo> weights)
    {
        int decoderStage = DecoderStage(tensors.Keys);
        foreach (var (name, tensor) in tensors)
        {
            if (tensor.Type is null)
            {
                continue;
            }

            var info = tensor.ToWeightInfo();
            if (name == "conditioner.conditioners.prompt.padding_embedding")
            {
                weights["text.padding"] = HfTransforms.ToF32(info, "text.padding");
            }
            else if (name.StartsWith("conditioner.conditioners.seconds_total.embedder.embedding.1.", StringComparison.Ordinal))
            {
                string canonical = "seconds.projection." + name[(name.LastIndexOf('.') + 1)..];
                weights[canonical] = HfTransforms.ToF32(info, canonical);
            }
            else if (name.StartsWith(Dit, StringComparison.Ordinal))
            {
                MapTransformer(name[Dit.Length..], info, weights);
            }
            else if (name == "pretransform.model.bottleneck.running_std")
            {
                weights["vae.latent_scale"] = HfTransforms.ToF32(info, "vae.latent_scale");
            }
            else if (name.StartsWith(Decoder, StringComparison.Ordinal))
            {
                MapDecoder(name[Decoder.Length..], info, tensors, decoderStage, weights);
            }
        }
    }

    /// <summary>Adds the encoder of the T5Gemma checkpoint under <c>text.</c> (the Gemma norms' <c>1 + w</c> folded, the layers' matrices f32).</summary>
    /// <param name="tensors">The tensors of the T5Gemma folder's <c>model.safetensors</c>.</param>
    /// <param name="weights">Receives the canonical weights.</param>
    public static void MapTextEncoder(IReadOnlyDictionary<string, SafetensorsTensor> tensors, Dictionary<string, WeightInfo> weights)
    {
        const string Encoder = "model.encoder.";
        foreach (var (name, tensor) in tensors)
        {
            if (!name.StartsWith(Encoder, StringComparison.Ordinal) || tensor.Type is null)
            {
                continue;
            }

            string local = name[Encoder.Length..];
            var info = tensor.ToWeightInfo();
            if (local == "embed_tokens.weight")
            {
                weights["text.token_embd.weight"] = HfTransforms.Plain(info, "text.token_embd.weight");
                continue;
            }

            if (local == "norm.weight")
            {
                weights["text.norm.weight"] = HfTransforms.NormPlusOne(info, "text.norm.weight");
                continue;
            }

            var m = TextLayer().Match(local);
            if (!m.Success)
            {
                throw new NotSupportedException($"T5Gemma encoder tensor '{name}' is not supported.");
            }

            string prefix = $"text.layers.{m.Groups[1].Value}.";
            string part = m.Groups[2].Value;
            string? norm = part switch
            {
                "pre_self_attn_layernorm.weight" => "attention_pre_norm.weight",
                "post_self_attn_layernorm.weight" => "attention_post_norm.weight",
                "pre_feedforward_layernorm.weight" => "mlp_pre_norm.weight",
                "post_feedforward_layernorm.weight" => "mlp_post_norm.weight",
                _ => null,
            };
            if (norm is not null)
            {
                weights[prefix + norm] = HfTransforms.NormPlusOne(info, prefix + norm);
                continue;
            }

            string canonical = prefix + part switch
            {
                "self_attn.q_proj.weight" => "attention.q.weight",
                "self_attn.k_proj.weight" => "attention.k.weight",
                "self_attn.v_proj.weight" => "attention.v.weight",
                "self_attn.o_proj.weight" => "attention.o.weight",
                "mlp.gate_proj.weight" => "mlp.gate.weight",
                "mlp.up_proj.weight" => "mlp.up.weight",
                "mlp.down_proj.weight" => "mlp.down.weight",
                _ => throw new NotSupportedException($"T5Gemma encoder tensor '{name}' is not supported."),
            };

            // f32 (the checkpoint is bf16): GPUs multiply bf16 weights with f16 activations, which costs the conditioning ~40 dB of precision that guidance amplifies.
            weights[canonical] = HfTransforms.ToF32(info, canonical);
        }
    }

    // The index of the decoder's (single) resampling stage among decoder.layers (after its input projection).
    private static int DecoderStage(IEnumerable<string> names)
    {
        var stages = names.Where(n => n.StartsWith(Decoder, StringComparison.Ordinal) && n.EndsWith(".new_tokens", StringComparison.Ordinal))
            .Select(n => int.Parse(n[Decoder.Length..n.IndexOf('.', Decoder.Length)], System.Globalization.CultureInfo.InvariantCulture)).Distinct().ToList();
        return stages.Count == 1 ? stages[0] : throw new NotSupportedException($"Only SAME decoders with one resampling stage are supported ({stages.Count} found).");
    }

    private static void MapTransformer(string name, WeightInfo info, Dictionary<string, WeightInfo> weights)
    {
        if (name.EndsWith(".inv_freq", StringComparison.Ordinal))
        {
            return;
        }

        if (name is "preprocess_conv.weight" or "postprocess_conv.weight")
        {
            // 1×1 convolutions [C, C, 1] as matrices [C, C].
            string canonical = name == "preprocess_conv.weight" ? "dit.input_mix.weight" : "dit.output_mix.weight";
            weights[canonical] = HfTransforms.ToF32(info, canonical, new Shape(info.Shape[0].Fixed, info.Shape[1].Fixed));
            return;
        }

        string? mapped = name switch
        {
            _ when Sequential(name, "to_timestep_embed.", "dit.time_embedding.") is { } s => s,
            _ when Sequential(name, "to_cond_embed.", "dit.condition_embedding.") is { } s => s,
            _ when Sequential(name, "to_global_embed.", "dit.global_embedding.") is { } s => s,
            _ when Sequential(name, "transformer.global_cond_embedder.", "dit.modulation_embedding.") is { } s => s,
            "transformer.memory_tokens" => "dit.memory_tokens",
            "transformer.project_in.weight" => "dit.proj_in.weight",
            "transformer.project_out.weight" => "dit.proj_out.weight",
            _ => null,
        };
        if (mapped is null && DitLayer().Match(name) is { Success: true } m)
        {
            string part = m.Groups[2].Value;
            string? local = part switch
            {
                "to_scale_shift_gate" => "modulation",
                "pre_norm.gamma" => "attention_norm.weight",
                "self_attn.to_qkv.weight" => "attention.qkv.weight",
                "self_attn.q_norm.gamma" => "attention.q_norm.weight",
                "self_attn.k_norm.gamma" => "attention.k_norm.weight",
                "self_attn.to_out.weight" => "attention.o.weight",
                "cross_attend_norm.gamma" => "cross_attention_norm.weight",
                "cross_attn.to_q.weight" => "cross_attention.q.weight",
                "cross_attn.to_kv.weight" => "cross_attention.kv.weight",
                "cross_attn.q_norm.gamma" => "cross_attention.q_norm.weight",
                "cross_attn.k_norm.gamma" => "cross_attention.k_norm.weight",
                "cross_attn.to_out.weight" => "cross_attention.o.weight",
                "ff_norm.gamma" => "mlp_norm.weight",
                "ff.ff.0.proj.weight" => "mlp.up.weight",
                "ff.ff.0.proj.bias" => "mlp.up.bias",
                "ff.ff.2.weight" => "mlp.down.weight",
                "ff.ff.2.bias" => "mlp.down.bias",
                _ => Sequential(part, "to_local_embed.", "local_embedding."),
            };
            mapped = local is null ? null : $"dit.layers.{m.Groups[1].Value}.{local}";
        }

        if (mapped is null)
        {
            throw new NotSupportedException($"Stable Audio 3 transformer tensor '{Dit}{name}' is not supported.");
        }

        weights[mapped] = info.Shape.Rank == 1 || mapped.EndsWith("memory_tokens", StringComparison.Ordinal)
            ? HfTransforms.ToF32(info, mapped)
            : HfTransforms.Plain(info, mapped);
    }

    private static void MapDecoder(string name, WeightInfo info, IReadOnlyDictionary<string, SafetensorsTensor> tensors, int stage, Dictionary<string, WeightInfo> weights)
    {
        string stagePrefix = $"{stage}.";
        if (!name.StartsWith(stagePrefix, StringComparison.Ordinal))
        {
            // The input projection (latents to the transformer width) precedes the stage: Transpose, Linear, Transpose, stage.
            string input = name == $"{stage - 2}.weight" ? "vae.decoder.in.weight"
                : name == $"{stage - 2}.bias" ? "vae.decoder.in.bias"
                : throw new NotSupportedException($"SAME decoder tensor '{Decoder}{name}' is not supported.");
            weights[input] = HfTransforms.ToF32(info, input);
            return;
        }

        string local = name[stagePrefix.Length..];
        switch (local)
        {
            case "new_tokens":
                weights["vae.decoder.new_tokens"] = HfTransforms.ToF32(info, "vae.decoder.new_tokens", new Shape(info.Shape[1].Fixed, info.Shape[2].Fixed));
                return;
            case "mapping.weight_g":
                weights["vae.decoder.out.weight"] = HfWeightNormFolding.Fold(info, tensors[Decoder + stagePrefix + "mapping.weight_v"].ToWeightInfo(), "vae.decoder.out.weight");
                return;
            case "mapping.weight_v":
                return;
            case "mapping.bias":
                weights["vae.decoder.out.bias"] = HfTransforms.ToF32(info, "vae.decoder.out.bias");
                return;
        }

        var m = DecoderLayer().Match(local);
        if (!m.Success)
        {
            throw new NotSupportedException($"SAME decoder tensor '{Decoder}{name}' is not supported.");
        }

        string part = m.Groups[2].Value;
        if (part.EndsWith(".inv_freq", StringComparison.Ordinal))
        {
            return;
        }

        string canonical = $"vae.decoder.layers.{m.Groups[1].Value}." + part switch
        {
            "self_attn.to_qkv.weight" => "attention.qkv.weight",
            "self_attn.to_out.weight" => "attention.o.weight",
            "ff.ff.0.proj.weight" => "mlp.up.weight",
            "ff.ff.0.proj.bias" => "mlp.up.bias",
            "ff.ff.2.weight" => "mlp.down.weight",
            "ff.ff.2.bias" => "mlp.down.bias",
            _ => DynamicTanh(part) ?? throw new NotSupportedException($"SAME decoder tensor '{Decoder}{name}' is not supported."),
        };
        weights[canonical] = info.Shape.Rank == 1 ? HfTransforms.ToF32(info, canonical) : HfTransforms.Plain(info, canonical);
    }

    // The DynamicTanh norms: alpha (a scalar), gamma (the scale) and beta (the shift).
    private static string? DynamicTanh(string part)
    {
        int dot = part.LastIndexOf('.');
        string module = part[..dot] switch
        {
            "pre_norm" => "attention_norm",
            "ff_norm" => "mlp_norm",
            "self_attn.q_norm" => "attention.q_norm",
            "self_attn.k_norm" => "attention.k_norm",
            _ => string.Empty,
        };
        string parameter = part[(dot + 1)..] switch
        {
            "alpha" => "alpha",
            "gamma" => "weight",
            "beta" => "bias",
            _ => string.Empty,
        };
        return module.Length == 0 || parameter.Length == 0 ? null : $"{module}.{parameter}";
    }

    // A torch.nn.Sequential of Linear, activation, Linear: index 0 is linear_1, index 2 linear_2.
    private static string? Sequential(string name, string module, string canonical)
    {
        if (!name.StartsWith(module, StringComparison.Ordinal))
        {
            return null;
        }

        string rest = name[module.Length..];
        return rest switch
        {
            "0.weight" => canonical + "linear_1.weight",
            "0.bias" => canonical + "linear_1.bias",
            "2.weight" => canonical + "linear_2.weight",
            "2.bias" => canonical + "linear_2.bias",
            _ => null,
        };
    }

    [GeneratedRegex(@"^layers\.(\d+)\.(.+)$")]
    private static partial Regex TextLayer();

    [GeneratedRegex(@"^transformer\.layers\.(\d+)\.(.+)$")]
    private static partial Regex DitLayer();

    [GeneratedRegex(@"^transformers\.(\d+)\.(.+)$")]
    private static partial Regex DecoderLayer();
}
