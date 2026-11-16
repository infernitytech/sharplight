using System.Text.RegularExpressions;
using Sharplight.Import.Gguf;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Translates an OpenCLIP text tower as stored in original SD 2.x checkpoints (<c>cond_stage_model.model.*</c>: <c>resblocks</c>
/// with fused q/k/v <c>in_proj</c>, <c>c_fc</c> / <c>c_proj</c> MLPs) into transformers' <c>CLIPTextModel</c> names, which
/// <see cref="DiffusersNames.AddClipText"/> maps further.
/// </summary>
internal static partial class OpenClipNames
{
    /// <summary>
    /// The text tower's tensors under transformers names. SD 2.x conditions on the penultimate layer (followed by the final
    /// LayerNorm), so the last <paramref name="droppedLayers"/> residual blocks are left out, as diffusers' conversion does.
    /// </summary>
    /// <param name="tensors">The checkpoint's tensors.</param>
    /// <param name="prefix">The tower's prefix (<c>cond_stage_model.model.</c>).</param>
    /// <param name="droppedLayers">Trailing residual blocks not used for the conditioning.</param>
    /// <param name="projection">
    /// Also yield the text projection as transformers' <c>text_projection.weight</c> (SDXL's pooled embedding; OpenCLIP stores the
    /// transposed matrix).
    /// </param>
    /// <exception cref="InvalidDataException">A tensor of the tower is not recognized.</exception>
    public static IEnumerable<(string Name, WeightInfo Weight)> ToTransformers(IReadOnlyDictionary<string, WeightInfo> tensors, string prefix,
        int droppedLayers, bool projection = false)
    {
        var names = tensors.Keys.Where(n => n.StartsWith(prefix, StringComparison.Ordinal)).ToList();
        int layers = names.Select(n => ResBlock().Match(n[prefix.Length..])).Where(m => m.Success)
            .Select(m => int.Parse(m.Groups[1].ValueSpan, System.Globalization.CultureInfo.InvariantCulture) + 1).DefaultIfEmpty(0).Max();
        int kept = Math.Max(1, layers - droppedLayers);
        foreach (string name in names)
        {
            string s = name[prefix.Length..];
            var weight = tensors[name];
            switch (s)
            {
                case "token_embedding.weight":
                    yield return ("text_model.embeddings.token_embedding.weight", weight);
                    continue;
                case "positional_embedding":
                    yield return ("text_model.embeddings.position_embedding.weight", weight);
                    continue;
                case "ln_final.weight" or "ln_final.bias":
                    yield return ("text_model.final_layer_norm." + s[(s.IndexOf('.') + 1)..], weight);
                    continue;
                case "text_projection":
                    if (projection)
                    {
                        yield return ("text_projection.weight", Transposed(weight, "text_projection.weight"));
                    }

                    continue;
                case "logit_scale":
                    continue;
            }

            var block = ResBlock().Match(s);
            if (!block.Success)
            {
                throw new InvalidDataException($"Unexpected OpenCLIP text tensor '{name}'.");
            }

            int layer = int.Parse(block.Groups[1].ValueSpan, System.Globalization.CultureInfo.InvariantCulture);
            if (layer >= kept)
            {
                continue;
            }

            string target = $"text_model.encoder.layers.{layer}.";
            string rest = block.Groups[2].Value;
            if (rest is "attn.in_proj_weight" or "attn.in_proj_bias")
            {
                string suffix = rest.EndsWith("weight", StringComparison.Ordinal) ? "weight" : "bias";
                long width = weight.Shape[0].Fixed / 3;
                if (weight.Shape.Rank == 1 && weight.Type.IsQuantized)
                {
                    // A block-quantized bias (stable-diffusion.cpp's Q4_K files) cannot be cut at the q/k/v boundaries block-wise.
                    weight = Decoded(weight);
                }

                yield return (target + "self_attn.q_proj." + suffix, weight.RowSlice(target + "self_attn.q_proj." + suffix, 0, width));
                yield return (target + "self_attn.k_proj." + suffix, weight.RowSlice(target + "self_attn.k_proj." + suffix, width, width));
                yield return (target + "self_attn.v_proj." + suffix, weight.RowSlice(target + "self_attn.v_proj." + suffix, 2 * width, width));
                continue;
            }

            int dot = rest.LastIndexOf('.');
            string module = rest[..dot] switch
            {
                "ln_1" => "layer_norm1",
                "ln_2" => "layer_norm2",
                "attn.out_proj" => "self_attn.out_proj",
                "mlp.c_fc" => "mlp.fc1",
                "mlp.c_proj" => "mlp.fc2",
                _ => throw new InvalidDataException($"Unexpected OpenCLIP text tensor '{name}'."),
            };
            yield return (target + module + rest[dot..], weight);
        }
    }

    /// <summary>
    /// The transpose of a 2-D weight, produced on first read. Element-wise storage types keep their type; a block-quantized matrix
    /// (stable-diffusion.cpp stores the projection as Q4_K in some files) is decoded and transposed as f32, since its blocks run along the rows.
    /// </summary>
    internal static WeightInfo Transposed(WeightInfo weight, string name)
    {
        long rows = weight.Shape[0].Fixed;
        long columns = weight.Shape[1].Fixed;
        var type = weight.Type.IsQuantized ? DataType.F32 : weight.Type;
        int element = (int)type.ByteSize(1);
        return WeightInfo.Derived(name, type, new Shape(columns, rows), () =>
        {
            var source = weight.Type.IsQuantized ? ToBytes(GgmlBlockDecoder.Decode(weight.Type, weight.ReadAll())) : weight.ReadAll();
            var result = new byte[source.Length];
            for (long r = 0; r < rows; r++)
            {
                for (long c = 0; c < columns; c++)
                {
                    Buffer.BlockCopy(source, (int)((r * columns + c) * element), result, (int)((c * rows + r) * element), element);
                }
            }

            return result;
        });
    }

    /// <summary>A block-quantized weight decoded to f32 on first read.</summary>
    private static WeightInfo Decoded(WeightInfo weight)
        => WeightInfo.Derived(weight.Name, DataType.F32, weight.Shape, () => ToBytes(GgmlBlockDecoder.Decode(weight.Type, weight.ReadAll())));

    private static byte[] ToBytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    [GeneratedRegex(@"^transformer\.resblocks\.(\d+)\.(.+)$")]
    private static partial Regex ResBlock();
}
