using Sharplight.ImageGeneration;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads transformer-only HunyuanVideo checkpoints in the original tensor names: GGUF files (city96, any ggml type) and single
/// <c>.safetensors</c> files (also with ComfyUI's <c>model.diffusion_model.</c> prefix). The language model, CLIP and the VAE come as auxiliary
/// sources (a HunyuanVideo diffusers folder, merged by the <c>HunyuanVideoCompanionAdapter</c>).
/// </summary>
internal static class HunyuanVideoCheckpoint
{
    private static readonly string[] s_prefixes = ["", "model.diffusion_model."];

    /// <summary>The prefix of the original-name transformer tensors, or null when the names are not a HunyuanVideo transformer's.</summary>
    /// <param name="names">The tensor names of the file.</param>
    public static string? TransformerPrefix(IReadOnlyCollection<string> names)
        => s_prefixes.FirstOrDefault(prefix => HunyuanVideoNames.IsOriginal(names, prefix));

    /// <summary>Opens the file as a transformer-only canonical <c>hunyuan_video</c> source (configured from the tensor shapes).</summary>
    /// <param name="path">The checkpoint.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            string prefix = TransformerPrefix([.. tensors.Tensors.Keys]) ?? throw new NotSupportedException($"'{path}' is not a HunyuanVideo transformer.");
            var weights = new CanonicalWeights();
            HunyuanVideoNames.AddOriginal(tensors.Tensors, prefix, weights);
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path),
                [CanonicalKeys.Architecture] = HunyuanVideoFolder.Architecture,
            };
            AddConfig(metadata, weights.Weights);
            HunyuanVideoFolder.AddDefaults(metadata, null);
            return new ModelSource
            {
                Format = CheckpointTensors.IsGguf(path) ? "gguf" : "safetensors",
                Architecture = HunyuanVideoFolder.Architecture,
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

    /// <summary>
    /// The transformer configuration from the tensor shapes (128-wide heads, patch 1×2×2, RoPE (16, 56, 56) with base 256, as in every
    /// published HunyuanVideo; a companion folder of the same width overrides the head layout).
    /// </summary>
    private static void AddConfig(Dictionary<string, object> metadata, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        const string w = "transformer.";
        string t = HunyuanVideoFolder.Prefix + "transformer.";
        var patch = weights[w + "img_in.proj.weight"].Shape;
        int hidden = (int)patch[0].Fixed;
        int Count(string start) => weights.Keys.Count(k => k.StartsWith(w + start, StringComparison.Ordinal) && k.EndsWith(".norm1.weight", StringComparison.Ordinal));
        metadata[t + "hidden_size"] = hidden;
        metadata[t + "in_channels"] = (int)(patch[1].Fixed / 4);
        metadata[t + "out_channels"] = (int)(weights[w + "final_layer.linear.weight"].Shape[0].Fixed / 4);
        metadata[t + "patch_size"] = 2;
        metadata[t + "patch_size_t"] = 1;
        metadata[t + "head_count"] = hidden / 128;
        metadata[t + "head_dim"] = 128;
        metadata[t + "double_layers"] = weights.Keys.Count(k => k.StartsWith(w + "double_blocks.", StringComparison.Ordinal) && k.EndsWith(".img_attn_qkv.weight", StringComparison.Ordinal));
        metadata[t + "single_layers"] = weights.Keys.Count(k => k.StartsWith(w + "single_blocks.", StringComparison.Ordinal) && k.EndsWith(".linear1.weight", StringComparison.Ordinal));
        metadata[t + "refiner_layers"] = Count("txt_in.individual_token_refiner.blocks.");
        metadata[t + "text_dim"] = (int)weights[w + "txt_in.input_embedder.weight"].Shape[1].Fixed;
        metadata[t + "pooled_dim"] = (int)weights[w + "vector_in.in_layer.weight"].Shape[1].Fixed;
        metadata[t + "guidance_embed"] = weights.ContainsKey(w + "guidance_in.mlp.0.weight");
        metadata[t + "rope_theta"] = 256f;
        metadata[t + "rope_axes_dim"] = new[] { 16, 56, 56 };
        metadata[HunyuanVideoFolder.Prefix + "scheduler.sampler"] = Samplers.Euler;
    }
}
