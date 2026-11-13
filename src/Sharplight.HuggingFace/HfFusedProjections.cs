using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>Lazy splits of fused attention and expert projections whose layout is not a plain concatenation of row blocks.</summary>
internal static class HfFusedProjections
{
    /// <summary>
    /// A weight made of the given rows of <paramref name="source"/> (viewed as [rows, rest...]), computed on first read.
    /// One-dimensional results (biases) are widened to F32 like every canonical 1-D weight.
    /// </summary>
    public static WeightInfo GatherRows(WeightInfo source, string name, int[] rows)
    {
        long total = source.Shape[0].Fixed;
        long rowBytes = source.Type.ByteSize(source.Shape.Elements() / total);
        var dims = new List<Dim> { rows.Length };
        for (int d = 1; d < source.Shape.Rank; d++)
        {
            dims.Add(source.Shape[d]);
        }

        var gathered = WeightInfo.Derived(name, source.Type, new Shape(dims), () =>
        {
            var all = source.ReadAll();
            var result = new byte[rows.Length * rowBytes];
            for (int i = 0; i < rows.Length; i++)
            {
                Buffer.BlockCopy(all, checked((int)(rows[i] * rowBytes)), result, checked((int)(i * rowBytes)), checked((int)rowBytes));
            }

            return result;
        });
        return gathered.Shape.Rank <= 1 ? HfTransforms.Plain(gathered, name) : gathered;
    }

    /// <summary>
    /// Splits a fused q/k/v tensor whose rows are grouped per key/value head — <c>[kvHeads][queriesPerKv + 2][headDim]</c>, the queries of
    /// the group first, then its key and its value (Falcon with the new decoder layout, InternLM2) — into canonical q, k and v weights.
    /// With one query per group this is the per-head interleaved <c>[heads][3][headDim]</c> layout of GPT-NeoX.
    /// </summary>
    /// <param name="fused">The fused weight or bias.</param>
    /// <param name="layerPrefix">Canonical prefix of the layer (<c>layers.N.</c>).</param>
    /// <param name="kind">"weight" or "bias".</param>
    /// <param name="heads">Number of query heads.</param>
    /// <param name="kvHeads">Number of key/value heads (groups).</param>
    /// <param name="headDim">Head size.</param>
    /// <param name="collector">Receives the three weights.</param>
    public static void SplitGroupedQkv(WeightInfo fused, string layerPrefix, string kind, int heads, int kvHeads, int headDim, HfWeightCollector collector)
    {
        int queriesPerKv = heads / kvHeads;
        int groupRows = (queriesPerKv + 2) * headDim;
        var queryRows = new List<int>();
        var keyRows = new List<int>();
        var valueRows = new List<int>();
        for (int group = 0; group < kvHeads; group++)
        {
            int start = group * groupRows;
            queryRows.AddRange(Enumerable.Range(start, queriesPerKv * headDim));
            keyRows.AddRange(Enumerable.Range(start + queriesPerKv * headDim, headDim));
            valueRows.AddRange(Enumerable.Range(start + (queriesPerKv + 1) * headDim, headDim));
        }

        collector.Add(GatherRows(fused, $"{layerPrefix}attention.q.{kind}", [.. queryRows]));
        collector.Add(GatherRows(fused, $"{layerPrefix}attention.k.{kind}", [.. keyRows]));
        collector.Add(GatherRows(fused, $"{layerPrefix}attention.v.{kind}", [.. valueRows]));
    }

    /// <summary>
    /// Splits the second axis of a rank-3 expert tensor [E, A + B, K] into [E, A, K] and [E, B, K] (fused gate+up experts, gate first),
    /// computed lazily. Only dense float storage types are supported.
    /// </summary>
    public static (WeightInfo First, WeightInfo Second) SplitSecondAxis(WeightInfo fused, string firstName, string secondName, long firstLength)
    {
        long experts = fused.Shape[0].Fixed;
        long rows = fused.Shape[1].Fixed;
        long columns = fused.Shape[2].Fixed;
        long rowBytes = fused.Type.ByteSize(columns);
        WeightInfo Part(string name, long firstRow, long length)
            => WeightInfo.Derived(name, fused.Type, new Shape(experts, length, columns), () =>
            {
                var all = fused.ReadAll();
                var result = new byte[experts * length * rowBytes];
                for (long e = 0; e < experts; e++)
                {
                    Buffer.BlockCopy(all, checked((int)((e * rows + firstRow) * rowBytes)), result, checked((int)(e * length * rowBytes)), checked((int)(length * rowBytes)));
                }

                return result;
            });

        return (Part(firstName, 0, firstLength), Part(secondName, firstLength, rows - firstLength));
    }
}
