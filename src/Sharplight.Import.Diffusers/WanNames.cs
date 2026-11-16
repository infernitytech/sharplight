using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Maps Wan checkpoints to the canonical names of the <c>wan</c> architecture: the transformer to the original Wan names (from
/// diffusers' <c>WanTransformer3DModel</c> names, or unchanged from original-name files), the VAE to diffusers' <c>AutoencoderKLWan</c>
/// names with the time taps of every causal 3-D kernel folded into its input channels.
/// </summary>
internal static class WanNames
{
    /// <summary>Canonical prefix of the transformer.</summary>
    public const string TransformerPrefix = "transformer.";

    /// <summary>Canonical prefix of the second (low-noise) transformer of a two-expert model.</summary>
    public const string SecondTransformerPrefix = "transformer_2.";

    /// <summary>Canonical prefix of the VAE.</summary>
    public const string VaePrefix = "vae.";

    /// <summary>Canonical prefix of the image encoder (Wan 2.1 I2V / FLF2V: CLIP ViT-H/14).</summary>
    public const string ImageEncoderPrefix = "image_encoder.";

    private static readonly (string Diffusers, string Original)[] s_transformerRenames =
    [
        ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
        ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
        ("condition_embedder.time_proj.", "time_projection.1."),
        ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
        ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
        ("proj_out.", "head.head."),
        ("condition_embedder.image_embedder.norm1.", "img_emb.proj.0."),
        ("condition_embedder.image_embedder.ff.net.0.proj.", "img_emb.proj.1."),
        ("condition_embedder.image_embedder.ff.net.2.", "img_emb.proj.3."),
        ("condition_embedder.image_embedder.norm2.", "img_emb.proj.4."),
        ("condition_embedder.image_embedder.pos_embed", "img_emb.emb_pos"),
    ];

    private static readonly Dictionary<string, string> s_imageEncoderModules = new(StringComparer.Ordinal)
    {
        ["layer_norm1"] = "input_norm",
        ["layer_norm2"] = "post_attention_norm",
        ["self_attn.q_proj"] = "attention.q",
        ["self_attn.k_proj"] = "attention.k",
        ["self_attn.v_proj"] = "attention.v",
        ["self_attn.out_proj"] = "attention.o",
        ["mlp.fc1"] = "mlp.up",
        ["mlp.fc2"] = "mlp.down",
    };

    private static readonly (string Diffusers, string Original)[] s_blockRenames =
    [
        ("attn1.to_q.", "self_attn.q."),
        ("attn1.to_k.", "self_attn.k."),
        ("attn1.to_v.", "self_attn.v."),
        ("attn1.to_out.0.", "self_attn.o."),
        ("attn1.norm_q.", "self_attn.norm_q."),
        ("attn1.norm_k.", "self_attn.norm_k."),
        ("attn2.to_q.", "cross_attn.q."),
        ("attn2.to_k.", "cross_attn.k."),
        ("attn2.to_v.", "cross_attn.v."),
        ("attn2.to_out.0.", "cross_attn.o."),
        ("attn2.norm_q.", "cross_attn.norm_q."),
        ("attn2.norm_k.", "cross_attn.norm_k."),
        ("attn2.add_k_proj.", "cross_attn.k_img."),
        ("attn2.add_v_proj.", "cross_attn.v_img."),
        ("attn2.norm_added_k.", "cross_attn.norm_k_img."),
        ("norm2.", "norm3."),
        ("ffn.net.0.proj.", "ffn.0."),
        ("ffn.net.2.", "ffn.2."),
    ];

    /// <summary>Adds a transformer tensor under its canonical name (diffusers or original names).</summary>
    /// <param name="name">The tensor name without the component prefix.</param>
    /// <param name="stored">The stored tensor.</param>
    /// <param name="weights">The canonical set.</param>
    /// <param name="prefix">The canonical prefix (<see cref="TransformerPrefix"/>, or <see cref="SecondTransformerPrefix"/> for a second expert).</param>
    public static void AddTransformer(string name, WeightInfo stored, CanonicalWeights weights, string prefix = TransformerPrefix)
    {
        string canonical = TransformerName(name);
        if (canonical.EndsWith("modulation", StringComparison.Ordinal))
        {
            weights.AddF32(prefix + canonical, stored.As(prefix + canonical, new Shape(stored.Shape.Elements())));
            return;
        }

        if (canonical == "img_emb.emb_pos")
        {
            // [1, rows, width] (diffusers) or ggml's [width, rows, 1] read as [1, rows, width].
            long width = stored.Shape[stored.Shape.Rank - 1].Fixed;
            weights.AddF32(prefix + canonical, stored.As(prefix + canonical, new Shape(stored.Shape.Elements() / width, width)));
            return;
        }

        if (canonical == "patch_embedding.weight")
        {
            long outputs = stored.Shape[0].Fixed;
            weights.Add(prefix + canonical, stored.As(prefix + canonical, new Shape(outputs, stored.Shape.Elements() / outputs)));
            return;
        }

        weights.Add(prefix + canonical, stored);
    }

    /// <summary>Adds a VAE tensor (diffusers names) under its canonical name.</summary>
    /// <param name="name">The tensor name without the component prefix.</param>
    /// <param name="stored">The stored tensor.</param>
    /// <param name="weights">The canonical set.</param>
    public static void AddVae(string name, WeightInfo stored, CanonicalWeights weights)
    {
        string canonical = VaePrefix + name;
        if (name.EndsWith(".gamma", StringComparison.Ordinal))
        {
            weights.AddF32(canonical, stored.As(canonical, new Shape(stored.Shape.Elements())));
        }
        else if (stored.Shape.Rank == 5)
        {
            weights.Add(canonical, Conv3dKernels.FoldTimeTaps(canonical, stored));
        }
        else if (name.Contains(".to_qkv.", StringComparison.Ordinal) || name.Contains(".proj.", StringComparison.Ordinal))
        {
            weights.AddLinear(canonical, stored);
        }
        else
        {
            weights.Add(canonical, stored);
        }
    }

    /// <summary>
    /// Adds a tensor of a transformers <c>CLIPVisionModel(WithProjection)</c> under the image tower names of the canonical image encoder,
    /// keeping the layers before the penultimate hidden states (<paramref name="layers"/>); the last layer, the output norm and the projection
    /// are not used by Wan.
    /// </summary>
    /// <param name="name">The tensor name without the component prefix.</param>
    /// <param name="stored">The stored tensor.</param>
    /// <param name="weights">The canonical set.</param>
    /// <param name="layers">The layers to keep.</param>
    public static void AddImageEncoder(string name, WeightInfo stored, CanonicalWeights weights, int layers)
    {
        const string p = ImageEncoderPrefix;
        switch (name)
        {
            case "vision_model.embeddings.class_embedding":
                weights.AddF32(p + "class_embedding", stored.As(p + "class_embedding", new Shape(stored.Shape.Elements())));
                return;
            case "vision_model.embeddings.patch_embedding.weight":
                long outputs = stored.Shape[0].Fixed;
                weights.Add(p + "patch_embed.weight", stored.As(p + "patch_embed.weight", new Shape(outputs, stored.Shape.Elements() / outputs)));
                return;
            case "vision_model.embeddings.patch_embedding.bias":
                weights.Add(p + "patch_embed.bias", stored);
                return;
            case "vision_model.embeddings.position_embedding.weight":
                weights.AddF32(p + "position_embedding.weight", stored);
                return;
            case "vision_model.pre_layrnorm.weight" or "vision_model.pre_layrnorm.bias":
                weights.Add(p + "embedding_norm." + name[(name.LastIndexOf('.') + 1)..], stored);
                return;
        }

        const string layerPrefix = "vision_model.encoder.layers.";
        if (!name.StartsWith(layerPrefix, StringComparison.Ordinal))
        {
            return;
        }

        int dot = name.IndexOf('.', layerPrefix.Length);
        int layer = int.Parse(name.AsSpan(layerPrefix.Length, dot - layerPrefix.Length), System.Globalization.CultureInfo.InvariantCulture);
        int last = name.LastIndexOf('.');
        if (layer < layers && s_imageEncoderModules.TryGetValue(name[(dot + 1)..last], out var module))
        {
            weights.Add($"{p}layers.{layer}.{module}.{name[(last + 1)..]}", stored);
        }
    }

    private static string TransformerName(string name)
    {
        if (name == "scale_shift_table")
        {
            return "head.modulation";
        }

        foreach (var (diffusers, original) in s_transformerRenames)
        {
            if (name.StartsWith(diffusers, StringComparison.Ordinal))
            {
                return original + name[diffusers.Length..];
            }
        }

        const string blocks = "blocks.";
        if (!name.StartsWith(blocks, StringComparison.Ordinal))
        {
            return name;
        }

        int dot = name.IndexOf('.', blocks.Length);
        string block = name[..(dot + 1)];
        string rest = name[(dot + 1)..];
        if (rest == "scale_shift_table")
        {
            return block + "modulation";
        }

        foreach (var (diffusers, original) in s_blockRenames)
        {
            if (rest.StartsWith(diffusers, StringComparison.Ordinal))
            {
                return block + original + rest[diffusers.Length..];
            }
        }

        return name;
    }
}
