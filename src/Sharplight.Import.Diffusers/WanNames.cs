using Sharplight.Import.Gguf;
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

    /// <summary>Canonical prefix of the VAE.</summary>
    public const string VaePrefix = "vae.";

    private static readonly (string Diffusers, string Original)[] s_transformerRenames =
    [
        ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
        ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
        ("condition_embedder.time_proj.", "time_projection.1."),
        ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
        ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
        ("proj_out.", "head.head."),
    ];

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
        ("norm2.", "norm3."),
        ("ffn.net.0.proj.", "ffn.0."),
        ("ffn.net.2.", "ffn.2."),
    ];

    /// <summary>Adds a transformer tensor under its canonical name (diffusers or original names).</summary>
    /// <param name="name">The tensor name without the component prefix.</param>
    /// <param name="stored">The stored tensor.</param>
    /// <param name="weights">The canonical set.</param>
    public static void AddTransformer(string name, WeightInfo stored, CanonicalWeights weights)
    {
        string canonical = TransformerName(name);
        if (canonical.EndsWith("modulation", StringComparison.Ordinal))
        {
            weights.AddF32(TransformerPrefix + canonical, stored.As(TransformerPrefix + canonical, new Shape(stored.Shape.Elements())));
            return;
        }

        if (canonical == "patch_embedding.weight")
        {
            long outputs = stored.Shape[0].Fixed;
            weights.Add(TransformerPrefix + canonical, stored.As(TransformerPrefix + canonical, new Shape(outputs, stored.Shape.Elements() / outputs)));
            return;
        }

        weights.Add(TransformerPrefix + canonical, stored);
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
            weights.Add(canonical, FoldTimeTaps(canonical, stored));
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

    /// <summary>
    /// A causal 3-D kernel [O, C, kt, kh, kw] as the 2-D kernel [O, kt·C, kh, kw] of the stacked frame windows (time-tap-major input
    /// channels); quantized kernels are decoded to f32 first.
    /// </summary>
    private static WeightInfo FoldTimeTaps(string name, WeightInfo stored)
    {
        long outputs = stored.Shape[0].Fixed;
        long channels = stored.Shape[1].Fixed;
        long taps = stored.Shape[2].Fixed;
        long kh = stored.Shape[3].Fixed;
        long kw = stored.Shape[4].Fixed;
        var shape = new Shape(outputs, taps * channels, kh, kw);
        if (taps == 1)
        {
            return stored.As(name, shape);
        }

        var type = stored.Type.IsQuantized ? DataType.F32 : stored.Type;
        return WeightInfo.Derived(name, type, shape, () =>
        {
            byte[] source = stored.Type.IsQuantized ? F32Bytes(stored) : stored.ReadAll();
            int block = (int)type.ByteSize(kh * kw);
            var folded = new byte[source.Length];
            for (long o = 0; o < outputs; o++)
            {
                for (long c = 0; c < channels; c++)
                {
                    for (long t = 0; t < taps; t++)
                    {
                        long from = ((o * channels + c) * taps + t) * block;
                        long to = ((o * taps + t) * channels + c) * block;
                        Array.Copy(source, from, folded, to, block);
                    }
                }
            }

            return folded;
        });
    }

    private static byte[] F32Bytes(WeightInfo stored)
    {
        var values = GgmlBlockDecoder.Decode(stored.Type, stored.ReadAll());
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
