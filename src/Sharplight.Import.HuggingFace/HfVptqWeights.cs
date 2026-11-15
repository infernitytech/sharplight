using System.Text.Json;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// VPTQ checkpoints (Microsoft vector post-training quantization, <c>quant_method vptq</c>): per-module settings in
/// <c>quantization_config.config_for_layers</c>, tensors <c>X.indices</c>, <c>X.centroids.weight</c>, optional residual
/// (<c>X.res_indices</c>, <c>X.res_centroids.weight</c>), outlier (<c>X.outlier_indices</c>, <c>X.outlier_centroids.weight</c>),
/// column permutation (<c>X.perm</c>) and column normalization (<c>X.weight_scale</c>, <c>X.weight_bias</c>) are folded into a dense
/// <c>X.weight</c> in the centroids' dtype while the weight is read.
/// </summary>
/// <remarks>
/// Vectors run along the output dimension: output row o, column g of codebook c takes element <c>o mod v</c> of centroid
/// <c>indices[c, o / v, g]</c> (v = <c>vector_lens[1]</c>), plus the residual centroid of the same position; outlier columns (their
/// own vectors of <c>vector_lens[0]</c>) come first, then the codebooks' columns in order; the permutation moves column p to
/// <c>perm[p]</c>; finally <c>value * weight_scale + weight_bias</c> per column. Packed indices (<c>is_indice_packed</c>) are a
/// little-endian bit stream per [codebook, vector row] of I32 words, each element <c>index | residual index &lt;&lt; index bits</c>.
/// Every operation is rounded to the centroids' dtype as VPTQ's PyTorch <c>dequant</c> does, so the values are bit-identical to it.
/// This is a dequantize-at-import fallback (no codebook lookup kernel); rows are decoded on demand.
/// </remarks>
internal static class HfVptqWeights
{
    /// <summary>The checkpoint tensors with VPTQ modules replaced by dense weights; <paramref name="tensors"/> itself when there are none.</summary>
    /// <exception cref="NotSupportedException">A module's settings or layout are not readable.</exception>
    public static IReadOnlyDictionary<string, SafetensorsTensor> Resolve(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        if (!config.TryGetQuantization("quantization_config", out var settings)
            || !settings.TryGetProperty("quant_method", out var method) || method.GetString() != "vptq")
        {
            return tensors;
        }

        var modules = tensors.Keys.Where(name => name.EndsWith(".indices", StringComparison.Ordinal) && !name.EndsWith("_indices", StringComparison.Ordinal))
            .Select(name => name[..^".indices".Length])
            .Where(module => tensors.ContainsKey(module + ".centroids.weight"))
            .ToList();
        if (modules.Count == 0)
        {
            return tensors;
        }

        if (!settings.TryGetProperty("config_for_layers", out var layers) || layers.ValueKind != JsonValueKind.Object)
        {
            throw new NotSupportedException("VPTQ checkpoint without config_for_layers.");
        }

        var result = new Dictionary<string, SafetensorsTensor>(tensors, StringComparer.Ordinal);
        foreach (string module in modules)
        {
            if (!layers.TryGetProperty(module, out var layer))
            {
                throw new NotSupportedException($"{module}: VPTQ module without an entry in config_for_layers.");
            }

            var parts = new HfVptqModule(module, layer, tensors);
            foreach (string suffix in (string[])[".indices", ".centroids.weight", ".res_indices", ".res_centroids.weight", ".outlier_indices",
                ".outlier_centroids.weight", ".perm", ".weight_scale", ".weight_bias"])
            {
                result.Remove(module + suffix);
            }

            result[module + ".weight"] = parts.ToTensor(tensors[module + ".indices"]);
        }

        return result;
    }

    /// <summary>A value rounded to <paramref name="type"/> (F16 or BF16, round to nearest even) and back to f32.</summary>
    internal static float Round(DataType type, float value)
    {
        if (type == DataType.F16)
        {
            return (float)(Half)value;
        }

        if (float.IsNaN(value))
        {
            return value;
        }

        uint bits = BitConverter.SingleToUInt32Bits(value);
        return BitConverter.UInt32BitsToSingle(((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16) << 16);
    }
}
