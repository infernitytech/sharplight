using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// ONNX Unique over the flattened input: unique values (ascending or in first-occurrence order), their first indices, the inverse
/// mapping and the counts. Publishes the number of unique values as the dynamic symbol of the outputs.
/// </summary>
internal sealed unsafe class UniqueOp(TRef x, TRef values, TRef firstIndices, TRef inverse, TRef counts, string countSymbol, bool sorted) : CpuOp
{
    public override string Name => "Unique";

    public override void Run(RunParameters p)
    {
        long total = x.Elements(p);
        var firstOf = new Dictionary<double, int>();
        var order = new List<double>();
        for (int i = 0; i < total; i++)
        {
            double v = OpHelpers.Read(x, i);
            if (firstOf.TryAdd(v, i))
            {
                order.Add(v);
            }
        }

        if (sorted)
        {
            order.Sort();
        }

        var slot = new Dictionary<double, int>(order.Count);
        for (int u = 0; u < order.Count; u++)
        {
            slot[order[u]] = u;
            Write(values, u, order[u]);
            ((int*)firstIndices.P)[u] = firstOf[order[u]];
            ((int*)counts.P)[u] = 0;
        }

        for (int i = 0; i < total; i++)
        {
            int u = slot[OpHelpers.Read(x, i)];
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
}
