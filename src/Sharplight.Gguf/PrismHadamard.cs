using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// PrismML "Hadamard" GGUF contract (Ternary-Bonsai-2): linear weights are stored in a rotated basis so that they are exactly
/// ternary; the producing runtime rotates the activations instead (<c>x' = H(s ∘ x)</c> per block of
/// <c>prism.hadamard.block_size</c> input columns, H the normalized Sylvester-Walsh-Hadamard matrix, s a ±1 sign vector chosen by
/// the input width). Because H is symmetric and orthogonal, the plain weight is <c>row ← s ∘ (H · row_stored)</c>.
/// The importer undoes the rotation lazily so canonical weights are ordinary dense f16 weights (the same
/// formula reconstructs the "inverse after lookup" token embedding rows, which a tied output head shares).
/// The rotation is exact up to f16 rounding; the price is that these tensors are no longer stored in their 2-bit packing.
/// </summary>
internal static class PrismHadamard
{
    public static bool Applies(IReadOnlyDictionary<string, object> md) => md.ContainsKey("prism.hadamard.version");

    public static void Apply(Dictionary<string, WeightInfo> weights, IReadOnlyDictionary<string, object> md, Func<string, string?> canonicalName)
    {
        int version = Convert.ToInt32(md["prism.hadamard.version"]);
        if (version is not (1 or 2))
        {
            throw new NotSupportedException($"prism.hadamard.version {version} is not supported.");
        }

        string transform = md.GetValueOrDefault("prism.hadamard.transform") as string ?? "normalized-sylvester-walsh-hadamard";
        string axis = md.GetValueOrDefault("prism.hadamard.axis") as string ?? "input-last-dimension";
        string signMode = md.GetValueOrDefault("prism.hadamard.sign_mode") as string ?? "explicit";
        if (transform != "normalized-sylvester-walsh-hadamard" || axis != "input-last-dimension" || signMode != "explicit")
        {
            throw new NotSupportedException($"Unsupported Hadamard contract (transform '{transform}', axis '{axis}', sign mode '{signMode}').");
        }

        int block = Convert.ToInt32(md["prism.hadamard.block_size"]);
        if (block <= 0 || (block & (block - 1)) != 0)
        {
            throw new InvalidDataException($"prism.hadamard.block_size {block} is not a power of two.");
        }

        var widths = ((Array)md["prism.hadamard.sign_widths"]).Cast<object>().Select(Convert.ToInt32).ToArray();
        var values = ((Array)md["prism.hadamard.sign_values"]).Cast<object>().Select(Convert.ToInt32).ToArray();
        if (widths.Sum() != values.Length)
        {
            throw new InvalidDataException("prism.hadamard.sign_values does not match sign_widths.");
        }

        var signs = new Dictionary<int, float[]>();
        int off = 0;
        foreach (int w in widths) { signs[w] = values.AsSpan(off, w).ToArray().Select(v => v < 0 ? -1f : 1f).ToArray(); off += w; }

        var names = new List<string>();
        foreach (var key in new[] { "prism.hadamard.weight_names", "prism.hadamard.inverse_weight_names" })
        {
            if (md.TryGetValue(key, out var v) && v is string[] arr)
            {
                names.AddRange(arr);
            }
        }

        foreach (var gguf in names)
        {
            if (canonicalName(gguf) is not { } canon || !weights.TryGetValue(canon, out var w))
            {
                continue;
            }

            int k = checked((int)w.Shape[w.Shape.Rank - 1].Fixed);
            if (!signs.TryGetValue(k, out var s))
            {
                throw new InvalidDataException($"No Hadamard sign vector for input width {k} ({gguf}).");
            }

            if (k % block != 0)
            {
                throw new InvalidDataException($"{gguf}: input width {k} is not a multiple of the Hadamard block {block}.");
            }

            weights[canon] = Rotate(w, s, block);
        }
    }

    private static WeightInfo Rotate(WeightInfo w, float[] sign, int block)
    {
        long elems = w.Shape.Elements();
        int k = sign.Length;
        long rows = elems / k;
        return WeightInfo.Derived(w.Name, DataType.F16, w.Shape, () =>
        {
            var src = w.ReadAll();
            var dst = new byte[elems * 2];
            float norm = 1f / MathF.Sqrt(block);
            Parallel.For(0, (int)Math.Min(rows, int.MaxValue), () => new float[k], (r, _, row) =>
            {
                Dequantize(w.Type, src, r * (long)k, k, row);
                for (int b = 0; b < k; b += block)
                {
                    Fwht(row.AsSpan(b, block));
                }

                var d = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Half>(dst.AsSpan((int)(r * k * 2), k * 2));
                for (int j = 0; j < k; j++)
                {
                    d[j] = (Half)(row[j] * norm * sign[j]);
                }

                return row;
            }, _ => { });
            return dst;
        });
    }

    private static void Fwht(Span<float> a)
    {
        for (int h = 1; h < a.Length; h <<= 1)
        {
            for (int i = 0; i < a.Length; i += h << 1)
            {
                for (int j = i; j < i + h; j++) { float x = a[j], y = a[j + h]; a[j] = x + y; a[j + h] = x - y; }
            }
        }
    }

    private static float H16(byte[] b, long o) => (float)BitConverter.ToHalf(b, (int)o);

    private static int Trit(byte b, int n) { int pw = 1; for (int i = 0; i < n; i++) { pw *= 3; } return ((((b * pw) & 0xFF) * 3) >> 8) - 1; }

    /// <summary>Managed reference decoder of the group-wise formats (plus plain float types) for one row of <paramref name="count"/> elements starting at element <paramref name="first"/>.</summary>
    private static void Dequantize(DataType t, byte[] src, long first, int count, float[] dst)
    {
        if (t == DataType.F32) { for (int i = 0; i < count; i++) { dst[i] = BitConverter.ToSingle(src, (int)((first + i) * 4)); } return; }
        if (t == DataType.F16) { for (int i = 0; i < count; i++) { dst[i] = H16(src, (first + i) * 2); } return; }
        if (t == DataType.BF16) { for (int i = 0; i < count; i++) { dst[i] = BitConverter.UInt32BitsToSingle((uint)BitConverter.ToUInt16(src, (int)((first + i) * 2)) << 16); } return; }
        long bs = t.BlockSize;
        if (first % bs != 0 || count % bs != 0)
        {
            throw new InvalidDataException("Hadamard rows must be block aligned.");
        }

        for (long blk = first / bs, end = (first + count) / bs; blk < end; blk++)
        {
            long o = blk * t.BlockBytes;
            int e0 = (int)((blk * bs) - first);
            if (t == DataType.Q1_0)
            {
                float d = H16(src, o);
                for (int j = 0; j < 128; j++)
                {
                    dst[e0 + j] = ((src[o + 2 + j / 8] >> (j % 8)) & 1) != 0 ? d : -d;
                }
            }
            else if (t == DataType.Q2_0 || t == DataType.Q2_0_G128)
            {
                float d = H16(src, o);
                for (int j = 0; j < bs; j++)
                {
                    dst[e0 + j] = (((src[o + 2 + j / 4] >> (2 * (j % 4))) & 3) - 1) * d;
                }
            }
            else if (t == DataType.Tern1_G128)
            {
                float d = H16(src, o + 26);
                int j = 0;
                for (int n = 0; n < 5; n++)
                {
                    for (int m = 0; m < 16; m++)
                    {
                        dst[e0 + j++] = Trit(src[o + m], n) * d;
                    }
                }

                for (int n = 0; n < 5; n++)
                {
                    for (int m = 0; m < 8; m++)
                    {
                        dst[e0 + j++] = Trit(src[o + 16 + m], n) * d;
                    }
                }

                for (int n = 0; n < 4; n++)
                {
                    for (int h = 0; h < 2; h++)
                    {
                        dst[e0 + j++] = Trit(src[o + 24 + h], n) * d;
                    }
                }
            }
            else
            {
                throw new NotSupportedException($"Hadamard-rotated weights of type {t} are not supported.");
            }
        }
    }
}
