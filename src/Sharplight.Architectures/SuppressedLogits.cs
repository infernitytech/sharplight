using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Token ranges a model never emits (canonical <c>logits.suppressed_ranges</c>, int[] of [start, end) pairs): their logits are pushed to the
/// lowest float by adding a constant row, so greedy and sampled decoding skip them (Chameleon's image tokens in text-only generation).
/// </summary>
internal static class SuppressedLogits
{
    /// <summary>Adds the suppression row to <paramref name="logits"/> ([rows, <paramref name="vocab"/>]); unchanged without ranges.</summary>
    /// <exception cref="InvalidDataException">A range is empty or lies outside the vocabulary.</exception>
    public static Value Apply(GraphBuilder b, Value logits, int vocab, int[] ranges)
    {
        if (ranges.Length == 0)
        {
            return logits;
        }

        var row = new float[vocab];
        for (int i = 0; i < ranges.Length; i += 2)
        {
            int start = ranges[i];
            int end = Math.Min(ranges[i + 1], vocab);
            if (start < 0 || start >= end)
            {
                throw new InvalidDataException($"logits.suppressed_ranges: [{ranges[i]}, {ranges[i + 1]}) is not a range inside the {vocab}-token vocabulary.");
            }

            Array.Fill(row, float.MinValue, start, end - start);
        }

        var bytes = new byte[vocab * sizeof(float)];
        Buffer.BlockCopy(row, 0, bytes, 0, bytes.Length);
        return b.Add(logits, b.Constant("suppressed_logits", DataType.F32, new Shape(vocab), bytes), "logits_suppressed");
    }
}
