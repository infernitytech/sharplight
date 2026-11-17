using Sharplight.ImageGeneration;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads single-file LTX-Video checkpoints in the original (Lightricks / ComfyUI) names: transformer-only GGUF files (QuantStack, any ggml
/// type) and <c>.safetensors</c> transformers (also with <c>model.diffusion_model.</c>), and stand-alone VAE files (<c>decoder.*</c> with
/// <c>per_channel_statistics</c>), mapped to the diffusers names of the canonical <c>ltx</c> source as diffusers' single-file conversion does.
/// Transformer-only files take the T5 encoder and the VAE as auxiliary sources (the <c>LtxCompanionAdapter</c>).
/// </summary>
internal static class LtxCheckpoint
{
    /// <summary>The noise levels of the distilled 13B models' 8-step first pass (Lightricks' published timesteps / 1000).</summary>
    public static readonly float[] DistilledSigmas = [1.0f, 0.993f, 0.987f, 0.981f, 0.975f, 0.909f, 0.725f, 0.00003f];

    private static readonly string[] s_prefixes = ["", "model.diffusion_model."];

    // diffusers' VAE_095_RENAME_DICT for the decoder: the original flat up_blocks list → mid block, upsamplers and up blocks.
    private static readonly (string Original, string Diffusers)[] s_decoderBlocks =
    [
        ("decoder.up_blocks.0.", "decoder.mid_block."),
        ("decoder.up_blocks.1.", "decoder.up_blocks.0.upsamplers.0."),
        ("decoder.up_blocks.2.", "decoder.up_blocks.0."),
        ("decoder.up_blocks.3.", "decoder.up_blocks.1.upsamplers.0."),
        ("decoder.up_blocks.4.", "decoder.up_blocks.1."),
        ("decoder.up_blocks.5.", "decoder.up_blocks.2.upsamplers.0."),
        ("decoder.up_blocks.6.", "decoder.up_blocks.2."),
        ("decoder.up_blocks.7.", "decoder.up_blocks.3.upsamplers.0."),
        ("decoder.up_blocks.8.", "decoder.up_blocks.3."),
    ];

    /// <summary>The prefix of the original-name transformer tensors, or null when the names are not an LTX-Video transformer's.</summary>
    /// <param name="names">The tensor names of the file.</param>
    public static string? TransformerPrefix(IReadOnlyCollection<string> names)
        => s_prefixes.FirstOrDefault(prefix => names.Contains(prefix + "patchify_proj.weight") && names.Contains(prefix + "adaln_single.linear.weight"));

    /// <summary>Whether the names are a stand-alone LTX-Video VAE with a timestep-conditioned decoder (0.9.5 and later).</summary>
    /// <param name="names">The tensor names of the file.</param>
    public static bool IsVae(IReadOnlyCollection<string> names)
        => names.Contains("decoder.last_scale_shift_table") && names.Contains("per_channel_statistics.mean-of-means");

    /// <summary>Opens a transformer-only file or a stand-alone VAE file as a canonical <c>ltx</c> source.</summary>
    /// <param name="path">The checkpoint.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            var names = tensors.Tensors.Keys.ToList();
            var weights = new CanonicalWeights();
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path),
                [CanonicalKeys.Architecture] = LtxFolder.Architecture,
            };
            if (TransformerPrefix(names) is { } prefix)
            {
                foreach (var (name, weight) in tensors.Tensors)
                {
                    if (name.StartsWith(prefix, StringComparison.Ordinal))
                    {
                        LtxFolder.AddTransformer(TransformerName(name[prefix.Length..]), weight, weights);
                    }
                }

                AddTransformerConfig(metadata, weights.Weights, Path.GetFileName(path).Contains("distilled", StringComparison.OrdinalIgnoreCase));
            }
            else if (IsVae(names))
            {
                foreach (var (name, weight) in tensors.Tensors)
                {
                    if (VaeName(name) is { } mapped)
                    {
                        LtxFolder.AddVae(mapped, weight, weights);
                    }
                }

                AddVaeConfig(metadata, weights.Weights);
            }
            else
            {
                throw new NotSupportedException($"'{path}' is neither an LTX-Video transformer nor an LTX-Video VAE.");
            }

            return new ModelSource
            {
                Format = CheckpointTensors.IsGguf(path) ? "gguf" : "safetensors",
                Architecture = LtxFolder.Architecture,
                Metadata = metadata,
                Weights = weights.Weights,
                OwnedResources = tensors.Resources,
                Path = path,
            };
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }

    /// <summary>diffusers' transformer renames: <c>patchify_proj</c> → <c>proj_in</c>, <c>adaln_single</c> → <c>time_embed</c>, q/k norms.</summary>
    private static string TransformerName(string name)
        => name.Replace("patchify_proj", "proj_in", StringComparison.Ordinal)
            .Replace("adaln_single", "time_embed", StringComparison.Ordinal)
            .Replace("q_norm", "norm_q", StringComparison.Ordinal)
            .Replace("k_norm", "norm_k", StringComparison.Ordinal);

    /// <summary>The diffusers name of a VAE decoder tensor (encoder tensors are not used), or null.</summary>
    private static string? VaeName(string name)
    {
        switch (name)
        {
            case "per_channel_statistics.mean-of-means":
                return "latents_mean";
            case "per_channel_statistics.std-of-means":
                return "latents_std";
        }

        if (!name.StartsWith("decoder.", StringComparison.Ordinal))
        {
            return null;
        }

        foreach (var (original, diffusers) in s_decoderBlocks)
        {
            if (name.StartsWith(original, StringComparison.Ordinal))
            {
                name = diffusers + name[original.Length..];
                break;
            }
        }

        return name.Replace("res_blocks", "resnets", StringComparison.Ordinal)
            .Replace("conv_shortcut.", "conv_shortcut.conv.", StringComparison.Ordinal)
            .Replace("norm3.norm", "norm3", StringComparison.Ordinal)
            .Replace("last_time_embedder", "time_embedder", StringComparison.Ordinal)
            .Replace("last_scale_shift_table", "scale_shift_table", StringComparison.Ordinal);
    }

    /// <summary>
    /// The transformer configuration from the tensor shapes (32 heads, as in every published LTX-Video; causal RoPE frame coordinates and a
    /// masked T5 as Lightricks' inference uses for single files), the pipeline defaults, and for distilled files (by name) their 8-step schedule without
    /// guidance.
    /// </summary>
    private static void AddTransformerConfig(Dictionary<string, object> metadata, IReadOnlyDictionary<string, WeightInfo> weights, bool distilled)
    {
        string t = LtxFolder.Prefix + "transformer.";
        int hidden = (int)weights["transformer.proj_in.weight"].Shape[0].Fixed;
        metadata[t + "latent_channels"] = (int)weights["transformer.proj_in.weight"].Shape[1].Fixed;
        metadata[t + "hidden_size"] = hidden;
        metadata[t + "head_count"] = 32;
        metadata[t + "layer_count"] = weights.Keys.Count(k => k.StartsWith("transformer.transformer_blocks.", StringComparison.Ordinal) && k.EndsWith(".attn1.to_q.weight", StringComparison.Ordinal));
        metadata[t + "caption_dim"] = (int)weights["transformer.caption_projection.linear_1.weight"].Shape[1].Fixed;
        metadata[t + "norm_eps"] = 1e-6f;
        metadata[t + "causal_frame_coordinates"] = true;
        metadata[LtxFolder.Prefix + "masked_text_encoder"] = true;
        metadata[LtxFolder.Prefix + "text_encoder.prompt_length"] = LtxFolder.PromptLength;
        metadata[LtxFolder.Prefix + "default_width"] = 704;
        metadata[LtxFolder.Prefix + "default_height"] = 480;
        metadata[LtxFolder.Prefix + "default_frames"] = 161;
        metadata[LtxFolder.Prefix + "frames_per_second"] = 25.0;
        metadata[LtxFolder.Prefix + "default_steps"] = 50;
        metadata[LtxFolder.Prefix + "default_guidance"] = 3.0;
        string s = LtxFolder.Prefix + "scheduler.";
        metadata[s + "sampler"] = Samplers.Euler;
        metadata[s + "train_steps"] = 1000;
        metadata[s + "dynamic_shift"] = false;
        metadata[s + "shift"] = 1f;
        metadata[s + "shift_terminal"] = 0f;
        if (distilled)
        {
            metadata[s + "sample_sigmas"] = DistilledSigmas;
            metadata[LtxFolder.Prefix + "default_steps"] = DistilledSigmas.Length;
            metadata[LtxFolder.Prefix + "default_guidance"] = 1.0;
        }
    }

    /// <summary>
    /// The VAE configuration of a timestep-conditioned decoder (0.9.5 and later): 4×4 pixel patches, ×8 in time and ×32 in space, residual
    /// upsamplers whose channel reduction follows from the kernel shapes.
    /// </summary>
    private static void AddVaeConfig(Dictionary<string, object> metadata, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        string v = LtxFolder.Prefix + "vae.";
        var factors = new List<int>();
        for (int i = 0; weights.TryGetValue($"vae.decoder.up_blocks.{i}.upsamplers.0.conv.conv.weight", out var conv); i++)
        {
            long input = conv.Shape[1].Fixed / 3;
            factors.Add((int)(input * 8 / conv.Shape[0].Fixed));
        }

        metadata[v + "latent_channels"] = (int)weights["vae.latents_mean"].Shape.Elements();
        metadata[v + "patch_size"] = 4;
        metadata[v + "spatial_downscale"] = 4 << factors.Count;
        metadata[v + "temporal_downscale"] = 1 << factors.Count;
        metadata[v + "upsample_residual"] = factors.Select(f => f > 1).ToArray();
        metadata[v + "upsample_factor"] = factors.ToArray();
        metadata[v + "decode_timestep"] = 0f;
    }
}
