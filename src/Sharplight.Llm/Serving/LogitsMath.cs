using System.Numerics;
using System.Runtime.InteropServices;

namespace Sharplight.Llm.Serving;

/// <summary>Vectorized helpers over a logits row.</summary>
internal static class LogitsMath
{
    /// <summary>Index of the largest value (the first one on ties). NaN entries never win; 0 when no entry is a number.</summary>
    public static int ArgMax(ReadOnlySpan<float> values)
    {
        float max = Max(values);
        int i = 0;
        if (Vector.IsHardwareAccelerated && values.Length >= Vector<float>.Count)
        {
            var target = new Vector<float>(max);
            var vectors = MemoryMarshal.Cast<float, Vector<float>>(values);
            int v = 0;
            while (v < vectors.Length && !Vector.EqualsAny(vectors[v], target))
            {
                v++;
            }

            i = v * Vector<float>.Count;
        }

        for (; i < values.Length; i++)
        {
            if (values[i] == max)
            {
                return i;
            }
        }

        return 0;
    }

    /// <summary>Largest value; NaN entries are skipped (negative infinity when there is no number).</summary>
    public static float Max(ReadOnlySpan<float> values)
    {
        float max = float.NegativeInfinity;
        int i = 0;
        if (Vector.IsHardwareAccelerated && values.Length >= Vector<float>.Count)
        {
            var acc = new Vector<float>(float.NegativeInfinity);
            var vectors = MemoryMarshal.Cast<float, Vector<float>>(values);
            foreach (var v in vectors)
            {
                // A comparison with NaN is false, so NaN lanes keep the accumulator.
                acc = Vector.ConditionalSelect(Vector.GreaterThan(v, acc), v, acc);
            }

            for (int lane = 0; lane < Vector<float>.Count; lane++)
            {
                max = Math.Max(max, acc[lane]);
            }

            i = vectors.Length * Vector<float>.Count;
        }

        for (; i < values.Length; i++)
        {
            if (values[i] > max)
            {
                max = values[i];
            }
        }

        return max;
    }
}
