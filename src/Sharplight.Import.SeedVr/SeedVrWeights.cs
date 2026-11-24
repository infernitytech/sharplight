using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.SeedVr;

/// <summary>
/// Maps the tensors of the SeedVR2 checkpoints onto the canonical names of the <c>seedvr</c> architecture: the DiT under
/// <c>transformer.</c> (branches <c>video</c>, <c>text</c> and <c>shared</c> for the layers whose weights both streams share), the VAE
/// under <c>vae.</c> with its 3-D convolution kernels folded to 2-D kernels over time-tap-major stacked frames, and the precomputed text
/// embeddings as <c>text_embedding.positive</c> / <c>text_embedding.negative</c>.
/// </summary>
internal static partial class SeedVrWeights
{
    /// <summary>The canonical name of the positive text embedding [L, text width].</summary>
    public const string PositiveEmbedding = "text_embedding.positive";

    /// <summary>The canonical name of the negative text embedding [L, text width].</summary>
    public const string NegativeEmbedding = "text_embedding.negative";

    private const string Transformer = "transformer.";
    private const string Vae = "vae.";

    private static readonly Dictionary<string, string> s_topLevel = new(StringComparer.Ordinal)
    {
        ["vid_in.proj.weight"] = "patch_in.weight",
        ["vid_in.proj.bias"] = "patch_in.bias",
        ["txt_in.weight"] = "text_in.weight",
        ["txt_in.bias"] = "text_in.bias",
        ["emb_in.proj_in.weight"] = "time_embed.linear_1.weight",
        ["emb_in.proj_in.bias"] = "time_embed.linear_1.bias",
        ["emb_in.proj_hid.weight"] = "time_embed.linear_2.weight",
        ["emb_in.proj_hid.bias"] = "time_embed.linear_2.bias",
        ["emb_in.proj_out.weight"] = "time_embed.linear_3.weight",
        ["emb_in.proj_out.bias"] = "time_embed.linear_3.bias",
        ["vid_out_norm.weight"] = "norm_out.weight",
        ["vid_out_ada.out_shift"] = "out_shift",
        ["vid_out_ada.out_scale"] = "out_scale",
        ["vid_out.proj.weight"] = "patch_out.weight",
        ["vid_out.proj.bias"] = "patch_out.bias",
    };

    private static readonly Dictionary<string, string> s_branches = new(StringComparer.Ordinal)
    {
        ["vid"] = "video",
        ["txt"] = "text",
        ["all"] = "shared",
    };

    private static readonly Dictionary<string, string> s_blockParts = new(StringComparer.Ordinal)
    {
        ["attn.proj_qkv"] = "qkv",
        ["attn.proj_out"] = "out",
        ["attn.norm_q"] = "q_norm",
        ["attn.norm_k"] = "k_norm",
        ["mlp.proj_in_gate"] = "mlp.gate",
        ["mlp.proj_in"] = "mlp.up",
        ["mlp.proj_out"] = "mlp.down",
    };

    /// <summary>Adds the DiT's tensors.</summary>
    /// <param name="tensors">The tensors of the DiT checkpoint.</param>
    /// <param name="weights">Receives the canonical weights.</param>
    /// <exception cref="NotSupportedException">A tensor has no canonical name (an unsupported variant).</exception>
    public static void MapTransformer(IReadOnlyDictionary<string, SafetensorsTensor> tensors, Dictionary<string, WeightInfo> weights)
    {
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            if (s_topLevel.TryGetValue(name, out string? top))
            {
                Add(weights, Transformer + top, info);
                continue;
            }

            var match = BlockName().Match(name);
            if (!match.Success)
            {
                throw new NotSupportedException($"SeedVR2 DiT tensor '{name}' has no canonical name.");
            }

            string block = Transformer + "blocks." + match.Groups["layer"].Value + ".";
            string part = match.Groups["part"].Success ? match.Groups["part"].Value
                : match.Groups["proj"].Success ? "attn." + match.Groups["proj"].Value
                : "mlp." + match.Groups["mlp"].Value;
            if (part == "attn.rope.rope")
            {
                // The frequency series of the 3-D RoPE, the same in every block.
                if (match.Groups["layer"].Value == "0")
                {
                    Add(weights, Transformer + "rope_frequencies", info);
                }

                continue;
            }

            string branch = s_branches[match.Groups["branch"].Value];
            string leaf = match.Groups["leaf"].Value;
            if (part == "ada")
            {
                Add(weights, block + branch + "." + leaf, info);
            }
            else if (s_blockParts.TryGetValue(part, out string? canonical))
            {
                Add(weights, block + branch + "." + canonical + "." + leaf, info);
            }
            else
            {
                throw new NotSupportedException($"SeedVR2 DiT tensor '{name}' has no canonical name.");
            }
        }
    }

    /// <summary>Adds the VAE's tensors under <c>vae.</c>, its 3-D kernels [O, C, kt, kh, kw] folded to [O, kt·C, kh, kw].</summary>
    /// <param name="tensors">The tensors of the VAE checkpoint.</param>
    /// <param name="weights">Receives the canonical weights.</param>
    public static void MapVae(IReadOnlyDictionary<string, SafetensorsTensor> tensors, Dictionary<string, WeightInfo> weights)
    {
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            string canonical = Vae + name;
            weights[canonical] = info.Shape.Rank == 5 ? FoldTimeTaps(canonical, info) : HfTransforms.Plain(info, canonical);
        }
    }

    /// <summary>Adds a precomputed text embedding [L, width] (a file holding one tensor) under <paramref name="canonical"/> in f32.</summary>
    /// <param name="tensors">The tensors of the embedding file.</param>
    /// <param name="canonical">The canonical name.</param>
    /// <param name="weights">Receives the weight.</param>
    /// <exception cref="InvalidDataException">The file does not hold one two-dimensional tensor.</exception>
    public static void MapEmbedding(IReadOnlyDictionary<string, SafetensorsTensor> tensors, string canonical, Dictionary<string, WeightInfo> weights)
    {
        if (tensors.Count != 1 || tensors.Values.First().Shape.Length != 2)
        {
            throw new InvalidDataException($"A SeedVR2 text embedding file holds one [tokens, width] tensor; found {tensors.Count} tensors.");
        }

        weights[canonical] = HfTransforms.ToF32(tensors.Values.First().ToWeightInfo(), canonical);
    }

    /// <summary>
    /// Vectors (biases, norms, modulation rows, frequencies) in f32; f32 matrices in f16 (the official model multiplies them in bf16 under
    /// autocast, so f16 loses nothing it keeps and halves the DiT's memory); other matrices as stored (f16, bf16, the fp8 conversions' E4M3).
    /// </summary>
    private static void Add(Dictionary<string, WeightInfo> weights, string canonical, WeightInfo info)
        => weights[canonical] = info.Shape.Rank <= 1 ? HfTransforms.ToF32(info, canonical)
            : info.Type == DataType.F32 ? ToF16(info, canonical)
            : info.As(canonical);

    /// <summary>An f32 tensor converted to f16 when it is read (round to nearest even).</summary>
    private static WeightInfo ToF16(WeightInfo source, string name)
        => WeightInfo.Derived(name, DataType.F16, source.Shape, () =>
        {
            byte[] bytes = source.ReadAll();
            int count = bytes.Length / sizeof(float);
            var halves = new byte[count * 2];
            const int Block = 1 << 16;
            Parallel.For(0, (count + Block - 1) / Block, block => ConvertBlock(bytes, halves, block * Block, Math.Min(count, (block + 1) * Block)));
            return halves;
        });

    private static void ConvertBlock(byte[] floats, byte[] halves, int start, int end)
    {
        var source = MemoryMarshal.Cast<byte, float>(floats.AsSpan(start * sizeof(float), (end - start) * sizeof(float)));
        var target = MemoryMarshal.Cast<byte, Half>(halves.AsSpan(start * 2, (end - start) * 2));
        for (int i = 0; i < source.Length; i++)
        {
            target[i] = (Half)source[i];
        }
    }

    /// <summary>
    /// A 3-D convolution kernel [O, C, kt, kh, kw] as the 2-D kernel [O, kt·C, kh, kw] of the frame windows stacked along the channels
    /// (time-tap-major), the form the VAE graphs convolve.
    /// </summary>
    private static WeightInfo FoldTimeTaps(string name, WeightInfo stored)
    {
        long outputs = stored.Shape[0].Fixed;
        long channels = stored.Shape[1].Fixed;
        long taps = stored.Shape[2].Fixed;
        long kh = stored.Shape[3].Fixed;
        long kw = stored.Shape[4].Fixed;
        var shape = new Shape(outputs, taps * channels, kh, kw);
        var source = stored.Type.IsQuantized ? HfTransforms.ToF32(stored, name) : stored;
        if (taps == 1)
        {
            return source.As(name, shape);
        }

        return WeightInfo.Derived(name, source.Type, shape, () =>
        {
            byte[] bytes = source.ReadAll();
            int block = (int)source.Type.ByteSize(kh * kw);
            var folded = new byte[bytes.Length];
            for (long o = 0; o < outputs; o++)
            {
                for (long c = 0; c < channels; c++)
                {
                    for (long t = 0; t < taps; t++)
                    {
                        long from = ((o * channels + c) * taps + t) * block;
                        long to = ((o * taps + t) * channels + c) * block;
                        Array.Copy(bytes, from, folded, to, block);
                    }
                }
            }

            return folded;
        });
    }

    [GeneratedRegex(@"^blocks\.(?<layer>\d+)\.(?:(?<part>attn\.rope\.rope)\.(?<leaf>freqs)|(?<part>ada)\.(?<branch>vid|txt|all)\.(?<leaf>\w+)"
        + @"|attn\.(?<proj>proj_qkv|proj_out|norm_q|norm_k)\.(?<branch>vid|txt|all)\.(?<leaf>weight|bias)"
        + @"|mlp\.(?<branch>vid|txt|all)\.(?<mlp>proj_in_gate|proj_in|proj_out)\.(?<leaf>weight|bias))$")]
    private static partial Regex BlockName();
}
