using System.Runtime.CompilerServices;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// ONNX Unique over the flattened input (<paramref name="sliceLength"/> 1) or over its slices along the outermost axis (each
/// <paramref name="sliceLength"/> elements long, compared lexicographically): unique elements (ascending or in first-occurrence order),
/// their first indices, the inverse mapping and the counts. Publishes the number of unique elements as the dynamic symbol of the outputs.
/// </summary>
internal sealed unsafe class UniqueOp(TRef x, TRef values, TRef firstIndices, TRef inverse, TRef counts, string countSymbol, bool sorted, long sliceLength)
    : CpuOp
{
    public override string Name => "Unique";

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public override void Run(RunParameters p)
    {
        int length = (int)sliceLength;
        long total = x.Elements(p) / length;
        var slices = new double[total][];
        for (int i = 0; i < total; i++)
        {
            var slice = new double[length];
            for (int j = 0; j < length; j++)
            {
                slice[j] = OpHelpers.Read(x, (long)i * length + j);
            }

            slices[i] = slice;
        }

        var firstOf = new Dictionary<double[], int>(SliceComparer.Instance);
        var order = new List<double[]>();
        for (int i = 0; i < total; i++)
        {
            if (firstOf.TryAdd(slices[i], i))
            {
                order.Add(slices[i]);
            }
        }

        if (sorted)
        {
            order.Sort(SliceComparer.Instance);
        }

        var slot = new Dictionary<double[], int>(order.Count, SliceComparer.Instance);
        for (int u = 0; u < order.Count; u++)
        {
            slot[order[u]] = u;
            for (int j = 0; j < length; j++)
            {
                Write(values, (long)u * length + j, order[u][j]);
            }

            ((int*)firstIndices.P)[u] = firstOf[order[u]];
            ((int*)counts.P)[u] = 0;
        }

        for (int i = 0; i < total; i++)
        {
            int u = slot[slices[i]];
            ((int*)inverse.P)[i] = u;
            ((int*)counts.P)[u]++;
        }

        p.Symbols[countSymbol] = order.Count;
    }

    private static void Write(TRef t, long i, double v)
    {
        if (t.Type == DataType.F32)
        {
            ((float*)t.P)[i] = (float)v;
        }
        else if (t.Type == DataType.I64)
        {
            ((long*)t.P)[i] = (long)v;
        }
        else
        {
            ((int*)t.P)[i] = (int)v;
        }
    }

    /// <summary>Element-wise equality and lexicographic order of equally long slices.</summary>
    private sealed class SliceComparer : IEqualityComparer<double[]>, IComparer<double[]>
    {
        public static readonly SliceComparer Instance = new();

        public bool Equals(double[]? a, double[]? b) => a.AsSpan().SequenceEqual(b);

        public int GetHashCode(double[] slice)
        {
            var hash = new HashCode();
            foreach (double v in slice)
            {
                hash.Add(v);
            }

            return hash.ToHashCode();
        }

        public int Compare(double[]? a, double[]? b) => a.AsSpan().SequenceCompareTo(b);
    }
}
