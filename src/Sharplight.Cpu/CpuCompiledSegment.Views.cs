using System.Collections;
using Sharplight.Graph;

namespace Sharplight.Cpu;

// Steps and zero-copy views: strided re-interpretation of existing buffers, with a materializing copy as fallback.
internal sealed unsafe partial class CpuCompiledSegment
{
    /// <summary>Appends an executable step and extends the lifetime of every arena buffer it touches.</summary>
    private void AddStep(CpuOp op, params TRef[] touches)
    {
        int index = _steps.Count;
        _steps.Add((op, touches));
        foreach (var t in touches)
        {
            if (!t.Buf.InArena)
            {
                continue;
            }

            if (index < t.Buf.First)
            {
                t.Buf.First = index;
            }

            if (index > t.Buf.Last)
            {
                t.Buf.Last = index;
            }
        }
    }

    /// <summary>The references an op touches, skipping absent optional ones (compile-time only).</summary>
    private static TRef[] Touched(params TRef?[] refs) => refs.Where(r => r is not null).ToArray()!;

    private TRef Materialize(TRef x)
    {
        var y = NewDense(x.Type, (long[])x.Dims.Clone(), x.Sym, "materialized");
        AddStep(new CopyOp(x, y), x, y);
        return y;
    }

    private TRef RowContig(TRef x) => x.LastContiguous ? x : Materialize(x);

    private TRef Dense(TRef x) => x.IsDense ? x : Materialize(x);

    private TRef View(TRef src, long[] dims, long[] strides, long offset, string? symbol) => Register(new TRef
    {
        Type = src.Type,
        Dims = dims,
        Strides = strides,
        Sym = symbol,
        Buf = src.Buf,
        Offset = src.Offset + offset,
    });

    /// <summary>
    /// Computes strides that reinterpret an (oldDims, oldStrides) view as newDims without copying (the NumPy
    /// attempt-nocopy-reshape algorithm). Returns false when the layout cannot be expressed as a view.
    /// </summary>
    private static bool TryReshapeStrides(long[] oldDims, long[] oldStrides, long[] newDims, out long[] newStrides)
    {
        newStrides = new long[newDims.Length];
        var keptDims = new List<long>();
        var keptStrides = new List<long>();
        for (int i = 0; i < oldDims.Length; i++)
        {
            if (oldDims[i] != 1)
            {
                keptDims.Add(oldDims[i]);
                keptStrides.Add(oldStrides[i]);
            }
        }

        int oldCount = keptDims.Count;
        int oldStart = 0;
        int oldEnd = 1;
        int newStart = 0;
        int newEnd = 1;
        while (newStart < newDims.Length && oldStart < oldCount)
        {
            long newProduct = newDims[newStart];
            long oldProduct = keptDims[oldStart];
            while (newProduct != oldProduct)
            {
                if (newProduct < oldProduct)
                {
                    if (newEnd >= newDims.Length)
                    {
                        return false;
                    }

                    newProduct *= newDims[newEnd++];
                }
                else
                {
                    if (oldEnd >= oldCount)
                    {
                        return false;
                    }

                    oldProduct *= keptDims[oldEnd++];
                }
            }

            for (int k = oldStart; k < oldEnd - 1; k++)
            {
                if (keptStrides[k] != keptDims[k + 1] * keptStrides[k + 1])
                {
                    return false;
                }
            }

            newStrides[newEnd - 1] = keptStrides[oldEnd - 1];
            for (int k = newEnd - 1; k > newStart; k--)
            {
                newStrides[k - 1] = newStrides[k] * newDims[k];
            }

            newStart = newEnd++;
            oldStart = oldEnd++;
        }

        long last = newStart >= 1 ? newStrides[newStart - 1] : 1;
        for (int k = newStart; k < newDims.Length; k++)
        {
            newStrides[k] = last;
        }

        return true;
    }

    private TRef ReshapeView(TRef source, Value output)
    {
        var newDims = DimsOf(output.Shape);
        string? symbol = output.Shape.Symbol;
        if (source.IsDense)
        {
            return View(source, newDims, TRef.RowMajor(newDims), 0, symbol);
        }

        long[] newStrides;
        if (source.Sym is not null && symbol is not null)
        {
            // The symbolic leading axis is kept as is; only the inner axes are reshaped.
            if (TryReshapeStrides(source.Dims[1..], source.Strides[1..], newDims[1..], out var inner))
            {
                newStrides = new long[newDims.Length];
                newStrides[0] = source.Strides[0];
                inner.CopyTo(newStrides, 1);
                return View(source, newDims, newStrides, 0, symbol);
            }
        }
        else if (source.Sym is null && symbol is null && TryReshapeStrides(source.Dims, source.Strides, newDims, out newStrides))
        {
            return View(source, newDims, newStrides, 0, symbol);
        }

        var copy = Materialize(source);
        return View(copy, newDims, TRef.RowMajor(newDims), 0, symbol);
    }

    private static int NormalizeAxis(long axis, int rank) => (int)(axis < 0 ? axis + rank : axis);

    private static int[] Ints(Attributes a, string key, params int[] defaults)
    {
        var values = IntArray(a, key);
        return values.Length > 0 ? values : defaults;
    }

    private static int[] IntArray(Attributes a, string key)
    {
        if (!a.TryGetValue(key, out var value))
        {
            return [];
        }

        return value switch
        {
            int[] ints => ints,
            long[] longs => longs.Select(x => (int)x).ToArray(),
            IEnumerable e => e.Cast<object>().Select(x => System.Convert.ToInt32(x)).ToArray(),
            _ => [],
        };
    }
}
