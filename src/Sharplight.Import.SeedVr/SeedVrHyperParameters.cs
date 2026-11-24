using Sharplight.Models;

namespace Sharplight.Import.SeedVr;

/// <summary>
/// Writes the canonical <c>seedvr.*</c> metadata of a SeedVR2 model: the DiT's sizes read from its tensor shapes, and the constants of
/// the official configurations (configs_3b / configs_7b main.yaml and the VAE's s8_c16_t4_inflation_sd3.yaml) that the checkpoints do
/// not carry: the window counts per axis and the 720p token grid they are laid out on, the RoPE scheme, the VAE's layout and latent
/// scaling factor, the one-step sampler's timestep and the colour correction of the official inference scripts.
/// </summary>
internal static class SeedVrHyperParameters
{
    private const string Prefix = SeedVrImporter.Architecture + ".";

    /// <summary>Adds the metadata of a model whose canonical weights are <paramref name="weights"/>.</summary>
    /// <param name="weights">The canonical weights (transformer, VAE and text embeddings).</param>
    /// <param name="metadata">Receives the keys.</param>
    /// <exception cref="InvalidDataException">The DiT lacks a tensor every SeedVR2 checkpoint has.</exception>
    public static void Add(IReadOnlyDictionary<string, WeightInfo> weights, Dictionary<string, object> metadata)
    {
        var patchIn = Require(weights, "transformer.patch_in.weight");
        var qkv = Require(weights, "transformer.blocks.0.video.qkv.weight");
        var queryNorm = Require(weights, "transformer.blocks.0.video.q_norm.weight");
        var textIn = Require(weights, "transformer.text_in.weight");
        var patchOut = Require(weights, "transformer.patch_out.weight");
        int layers = 0;
        while (weights.ContainsKey($"transformer.blocks.{layers}.video.qkv.weight") || weights.ContainsKey($"transformer.blocks.{layers}.shared.qkv.weight"))
        {
            layers++;
        }

        const int PatchArea = 4;
        long dim = patchIn.Shape[0].Fixed;
        long headDim = queryNorm.Shape[0].Fixed;
        string t = Prefix + "transformer.";
        metadata[t + "dim"] = dim;
        metadata[t + "head_dim"] = headDim;
        metadata[t + "heads"] = qkv.Shape[0].Fixed / 3 / headDim;
        metadata[t + "layers"] = (long)layers;
        metadata[t + "text_dim"] = textIn.Shape[1].Fixed;
        metadata[t + "in_channels"] = patchIn.Shape[1].Fixed / PatchArea;
        metadata[t + "out_channels"] = patchOut.Shape[0].Fixed / PatchArea;
        metadata[t + "patch_size"] = new long[] { 1, 2, 2 };
        metadata[t + "norm_eps"] = 1e-5;
        metadata[t + "time_embed_dim"] = 256L;
        bool gated = weights.ContainsKey("transformer.blocks.0.video.mlp.gate.weight");
        metadata[t + "mlp"] = gated ? "swiglu" : "gelu_tanh";
        // 3B (dit_v2): RoPE over the text too, the video's frame positions offset by the text length; 7B (dit): video only, positions
        // spread over [-1, 1] across each window.
        metadata[t + "rope"] = gated ? "text_offset" : "normalized";
        // 3B: the last block neither modulates the text stream nor runs its feed-forward (the text only provides keys and values there).
        metadata[t + "last_layer_text_passthrough"] = gated;
        metadata[t + "window_counts"] = new long[] { 4, 3, 3 };
        metadata[t + "window_reference_grid"] = new long[] { 45, 80 };
        metadata[t + "window_max_frames"] = 30L;

        string v = Prefix + "vae.";
        metadata[v + "block_channels"] = new long[] { 128, 256, 512, 512 };
        metadata[v + "layers_per_block"] = 2L;
        metadata[v + "latent_channels"] = 16L;
        metadata[v + "groups"] = 32L;
        metadata[v + "eps"] = 1e-6;
        metadata[v + "scaling_factor"] = 0.9152;
        metadata[v + "temporal_downscale"] = 4L;
        metadata[v + "spatial_downscale"] = 8L;

        metadata[Prefix + "timestep"] = 1000.0;
        metadata[Prefix + "color_correction"] = "wavelet";
        metadata[Prefix + "default_scale"] = 2.0;
    }

    private static WeightInfo Require(IReadOnlyDictionary<string, WeightInfo> weights, string name)
        => weights.TryGetValue(name, out var weight) ? weight : throw new InvalidDataException($"The SeedVR2 DiT has no '{name}' tensor.");
}
