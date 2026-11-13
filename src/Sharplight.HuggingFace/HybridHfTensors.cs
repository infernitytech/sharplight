using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>Weight conversions shared by the Hugging Face mappings onto the canonical <c>hybrid</c> architecture (documented on <c>HybridAdapter</c>).</summary>
internal static class HybridHfTensors
{
    /// <summary>
    /// Maps one Mamba-2 mixer tensor (name relative to the mixer module, e.g. <c>in_proj.weight</c>) of layer prefix <paramref name="prefix"/>
    /// (<c>layers.N.</c>); false when it is not a Mamba-2 tensor. A = -exp(A_log); the small parameters become F32.
    /// </summary>
    /// <param name="prefix">Canonical layer prefix including the trailing dot.</param>
    /// <param name="mixerSuffix">Tensor name relative to the mixer module.</param>
    /// <param name="w">The checkpoint tensor.</param>
    /// <param name="collector">Receives the canonical weight.</param>
    /// <param name="inProjection">Optional replacement for the input projection (e.g. with folded multipliers); receives the canonical name.</param>
    /// <param name="outProjectionScale">Factor folded into the output projection.</param>
    public static bool TryMapSsm(string prefix, string mixerSuffix, WeightInfo w, HfWeightCollector collector,
        Func<WeightInfo, string, WeightInfo>? inProjection = null, float outProjectionScale = 1f)
    {
        switch (mixerSuffix)
        {
            case "in_proj.weight":
                collector.Add(inProjection is null ? HfTransforms.Plain(w, prefix + "ssm.in.weight") : inProjection(w, prefix + "ssm.in.weight"));
                return true;
            case "out_proj.weight":
                collector.Add(outProjectionScale == 1f ? HfTransforms.Plain(w, prefix + "ssm.out.weight") : Scaled(w, prefix + "ssm.out.weight", outProjectionScale));
                return true;
            case "D":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.d"));
                return true;
            case "dt_bias":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.dt_bias"));
                return true;
            case "conv1d.bias":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.conv1d.bias"));
                return true;
            case "norm.weight":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.norm.weight"));
                return true;
            case "A_log":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.a", map: static v => -MathF.Exp(v)));
                return true;
            case "conv1d.weight":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.conv1d.weight", new Shape(w.Shape[0].Fixed, w.Shape[w.Shape.Rank - 1].Fixed)));
                return true;
            default:
                return false;
        }
    }

    /// <summary>The weight multiplied by a constant, as F32.</summary>
    public static WeightInfo Scaled(WeightInfo w, string name, float factor) => HfTransforms.ToF32(w, name, map: v => v * factor);

    /// <summary>The rows of a [rows, columns] weight multiplied by <paramref name="rowFactor"/>(row), as F32.</summary>
    public static WeightInfo ScaledRows(WeightInfo w, string name, Func<long, float> rowFactor)
    {
        long columns = w.Shape.Elements() / w.Shape[0].Fixed;
        return WeightInfo.Derived(name, DataType.F32, w.Shape, () =>
        {
            var values = HfTransforms.DecodeF32(w);
            for (long i = 0; i < values.LongLength; i++)
            {
                values[i] *= rowFactor(i / columns);
            }

            return HfTransforms.EncodeF32(values);
        });
    }
}
