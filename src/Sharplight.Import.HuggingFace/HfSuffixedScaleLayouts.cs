namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Quantized parameters named with an underscore instead of a module path (ModelOpt <c>parameter_groups</c> exports of fused expert banks,
/// thinkingmachines/Inkling-2-Mini): the codes are <c>P_weight</c> next to <c>P_weight_scale</c> and, for NVFP4, the per-matrix factor
/// <c>P_weight_amax</c> / <c>P_weight_outer_scale_denominator</c>. They are renamed to <c>P.weight</c>, <c>P.weight_scale</c>,
/// <c>P.weight_amax</c> and <c>P.weight_outer_scale_denominator</c>, so the FP8 (<see cref="HfFp8Weights"/>, MXFP8 with E8M0 exponents) and
/// FP4 (<see cref="HfFp4Weights"/>) readers convert them; mappings read the converted weight as <c>P.weight</c>.
/// </summary>
internal static class HfSuffixedScaleLayouts
{
    private const string CodesSuffix = "_weight";

    /// <summary>The companions of a quantized parameter, after the codes' name.</summary>
    private static readonly string[] s_companions = ["_scale", "_amax", "_outer_scale_denominator"];

    /// <summary>The checkpoint tensors with such parameters renamed (the others unchanged); <paramref name="tensors"/> itself when there are none.</summary>
    /// <param name="tensors">The checkpoint tensors.</param>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Rename(IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var stems = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (name, codes) in tensors)
        {
            if (name.EndsWith(CodesSuffix, StringComparison.Ordinal) && codes.DType is "U8" or "F8_E4M3"
                && tensors.ContainsKey(name + s_companions[0]))
            {
                stems.Add(name[..^CodesSuffix.Length]);
            }
        }

        if (stems.Count == 0)
        {
            return tensors;
        }

        var result = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            string renamed = name;
            foreach (string stem in stems)
            {
                if (!name.StartsWith(stem + CodesSuffix, StringComparison.Ordinal))
                {
                    continue;
                }

                string rest = name[(stem.Length + CodesSuffix.Length)..];
                if (rest.Length == 0 || s_companions.Contains(rest))
                {
                    renamed = stem + ".weight" + rest;
                }
            }

            result[renamed] = renamed == name ? tensor : HfQuantoLayouts.Renamed(tensor, renamed);
        }

        return result;
    }
}
