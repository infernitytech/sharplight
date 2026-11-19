using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Maps the decoder of a diffusers <c>AutoencoderOobleck</c> checkpoint (the audio autoencoder of Stable Audio Open and ACE-Step) onto canonical weights: the
/// weight-normalized convolutions are folded (<c>weight_g</c>, <c>weight_v</c> to <c>weight</c>), the snake parameters <c>[1, C, 1]</c> are flattened to <c>[C]</c>, every tensor
/// is f32. The encoder is not mapped.
/// </summary>
internal static class OobleckDecoderWeights
{
    private const string Gain = ".weight_g";
    private const string Direction = ".weight_v";

    /// <summary>Adds the decoder's tensors (<c>decoder.*</c>) under <paramref name="prefix"/> (<c>vae.</c> gives <c>vae.decoder.*</c>).</summary>
    /// <param name="tensors">The tensors of the autoencoder's safetensors file.</param>
    /// <param name="prefix">The canonical prefix.</param>
    /// <param name="add">Receives each canonical weight.</param>
    public static void Map(IReadOnlyDictionary<string, SafetensorsTensor> tensors, string prefix, Action<WeightInfo> add)
    {
        foreach (var (name, tensor) in tensors)
        {
            if (!name.StartsWith("decoder.", StringComparison.Ordinal) || name.EndsWith(Direction, StringComparison.Ordinal) || tensor.Type is null)
            {
                continue;
            }

            string canonical = prefix + name;
            var info = tensor.ToWeightInfo();
            if (name.EndsWith(Gain, StringComparison.Ordinal))
            {
                string module = name[..^Gain.Length];
                add(HfWeightNormFolding.Fold(info, tensors[module + Direction].ToWeightInfo(), prefix + module + ".weight"));
            }
            else if (name.EndsWith(".alpha", StringComparison.Ordinal) || name.EndsWith(".beta", StringComparison.Ordinal))
            {
                add(HfTransforms.ToF32(info, canonical, new Shape(info.Shape[1].Fixed)));
            }
            else
            {
                add(HfTransforms.ToF32(info, canonical));
            }
        }
    }
}
