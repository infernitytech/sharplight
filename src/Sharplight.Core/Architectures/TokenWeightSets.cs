using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Per-token weight sets: a layer's linear map <c>{prefix}{module}</c> with a second weight set <c>{prefix}media.{module}</c> (CogVLM's visual
/// experts) is applied with both sets and the rows are picked by a token-type mask ([T, 1], 1 = media weights, 0 = text weights, see
/// <see cref="InputRole.MediaWeightMask"/>): y = (x·Wtextᵀ + btext)·(1 − mask) + (x·Wmediaᵀ + bmedia)·mask. Without a mask (graphs that only
/// see text) or without media weights the map is the plain text one. Usable by any adapter.
/// </summary>
internal static class TokenWeightSets
{
    /// <summary>Infix of the media weight set below a layer prefix (<c>layers.{i}.media.attention.q.weight</c>).</summary>
    public const string MediaInfix = "media.";

    /// <summary>Name of the mask input.</summary>
    public const string MaskInput = "media_weight_mask";

    /// <summary>Whether any of the first <paramref name="layers"/> layers has a media weight set for <paramref name="modules"/>.</summary>
    public static bool Present(GraphBuilder b, int layers, params string[] modules)
        => Enumerable.Range(0, layers).Any(il => modules.Any(m => b.HasWeight($"layers.{il}.{MediaInfix}{m}.weight")));

    /// <summary>Declares the mask input [<paramref name="tokens"/>, 1] f32.</summary>
    public static Value DeclareMask(GraphBuilder b, Dim tokens) => b.Input(MaskInput, InputRole.MediaWeightMask, DataType.F32, new Shape(tokens, 1));

    /// <summary>
    /// The linear map <c>{prefix}{module}</c> (weight, optional bias) of <paramref name="x"/>, mixed per row with the media set when
    /// <paramref name="mask"/> is given and the layer has one.
    /// </summary>
    public static Value Linear(GraphBuilder b, Value x, Value? mask, string prefix, string module, string? name = null)
    {
        string media = prefix + MediaInfix + module;
        if (mask is null || !b.HasWeight(media + ".weight"))
        {
            return b.Linear(x, prefix + module + ".weight", prefix + module + ".bias", name);
        }

        return Mix(b, b.Linear(x, prefix + module + ".weight", prefix + module + ".bias"), b.Linear(x, media + ".weight", media + ".bias"), mask, name);
    }

    /// <summary>Whether the layer <paramref name="prefix"/> has a media weight set for <paramref name="module"/>.</summary>
    public static bool Has(GraphBuilder b, string prefix, string module) => b.HasWeight(prefix + MediaInfix + module + ".weight");

    /// <summary>Picks every row from <paramref name="text"/> or <paramref name="media"/> by <paramref name="mask"/> (1 = media).</summary>
    public static Value Mix(GraphBuilder b, Value text, Value media, Value mask, string? name = null)
        => b.Add(b.Mul(text, b.Scale(mask, -1f, 1f)), b.Mul(media, mask), name);
}
