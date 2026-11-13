using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

// Buffers and value references: how graph values map onto TRef views.
internal sealed unsafe partial class CpuCompiledSegment
{
    private static long ElementCount(long[] dims)
    {
        long elements = 1;
        foreach (var d in dims)
        {
            elements *= d;
        }

        return elements;
    }

    /// <summary>Upper bound for a symbolic dimension; remembers the largest bound so Execute can reject larger runs.</summary>
    private long Bound(string symbol)
    {
        long bound = symbol == WellKnown.ContextSymbol ? _options.ContextLength
                   : symbol == WellKnown.TokenSymbol ? _options.MaxTokens
                   : _options.SymbolBounds.TryGetValue(symbol, out var configured) ? configured : _options.MaxTokens;
        if (!_symbolBounds.TryGetValue(symbol, out var previous) || previous < bound)
        {
            _symbolBounds[symbol] = bound;
        }

        return bound;
    }

    /// <summary>Compile-time dimensions of a shape, with symbolic dimensions replaced by their bound.</summary>
    private long[] DimsOf(Shape shape)
    {
        var dims = new long[shape.Rank];
        for (int i = 0; i < dims.Length; i++)
        {
            dims[i] = shape.Dims[i].IsSymbolic ? Bound(shape.Dims[i].Symbol!) : shape.Dims[i].Fixed;
        }

        return dims;
    }

    private TRef Register(TRef r)
    {
        _allRefs.Add(r);
        return r;
    }

    /// <summary>Allocates native memory owned (and freed) by this segment.</summary>
    private byte* AllocateOwned(long bytes, bool zero)
    {
        byte* ptr = Mem.Alloc(bytes, zero);
        _owned.Add((nint)ptr);
        _ownedBytes += bytes;
        return ptr;
    }

    private Buf NewArenaBuf(long bytes, string name)
    {
        var buf = new Buf { Bytes = (Math.Max(bytes, 1) + 63) & ~63L, InArena = true, Name = name };
        _bufs.Add(buf);
        return buf;
    }

    private TRef NewDense(Value v)
    {
        if (v.Type.IsQuantized)
        {
            throw new NotSupportedException($"Quantized activation {v}.");
        }

        return NewDense(v.Type, DimsOf(v.Shape), v.Shape.Symbol, v.Name);
    }

    private TRef NewDense(DataType type, long[] dims, string? symbol, string name)
    {
        long elems = ElementCount(dims);
        return Register(new TRef
        {
            Type = type,
            Dims = dims,
            Strides = TRef.RowMajor(dims),
            Sym = symbol,
            Buf = NewArenaBuf(elems * type.BlockBytes, name),
        });
    }

    /// <summary>Makes the weight bytes addressable: used in place when the source is memory mapped, otherwise read into owned memory.</summary>
    private Buf LoadWeight(WeightInfo w)
    {
        if (_weightBufs.TryGetValue(w, out var cached))
        {
            return cached;
        }

        long bytes = w.ByteSize;
        var source = w.Source;
        var buf = new Buf { Bytes = bytes, Name = w.Name };
        nint mapped = source.TryGetPointer();
        if (mapped != 0)
        {
            buf.Ptr = (byte*)mapped;
        }
        else
        {
            buf.Ptr = AllocateOwned(bytes, false);
            const long chunk = 1L << 30;
            for (long offset = 0; offset < bytes; offset += chunk)
            {
                source.Read(offset, new Span<byte>(buf.Ptr + offset, (int)Math.Min(chunk, bytes - offset)));
            }
        }

        _weightBytes += bytes;
        _weightBufs[w] = buf;
        return buf;
    }

    private static bool IsStatic(Value v) => v.Kind is ValueKind.Weight or ValueKind.Constant;

    /// <summary>The value as seen by an op: weights/constants keep their raw storage type.</summary>
    private TRef Ref(Value v)
    {
        if (_refs.TryGetValue(v, out var r))
        {
            return r;
        }

        switch (v.Kind)
        {
            case ValueKind.Weight:
            {
                var dims = DimsOf(v.Shape);
                r = Register(new TRef { Type = v.Type, Dims = dims, Strides = TRef.RowMajor(dims), Buf = LoadWeight(v.Weight!) });
                break;
            }
            case ValueKind.Constant:
            {
                var data = v.ConstantData!;
                var buf = new Buf { Bytes = data.Length, Ptr = AllocateOwned(data.Length, false), Name = v.Name };
                data.CopyTo(new Span<byte>(buf.Ptr, data.Length));
                var dims = DimsOf(v.Shape);
                r = Register(new TRef { Type = v.Type, Dims = dims, Strides = TRef.RowMajor(dims), Buf = buf });
                break;
            }
            case ValueKind.State:
            {
                var def = v.State!;
                var dims = DimsOf(def.Shape);
                long bytes = def.Type.ByteSize(ElementCount(dims));
                var buf = new Buf { Bytes = bytes, Ptr = AllocateOwned(bytes, true), Name = def.Name };
                _stateBufs.Add((buf, bytes));
                r = Register(new TRef { Type = def.Type, Dims = dims, Strides = TRef.RowMajor(dims), Buf = buf });
                break;
            }
            default:
                // Input, or Intermediate produced outside this segment: dense buffer filled by SetInput.
                if (v.Kind == ValueKind.Intermediate && v.Producer is { } producer && _segment.Nodes.Contains(producer))
                {
                    throw new InvalidOperationException($"Value {v} used before it is produced.");
                }

                r = NewDense(v);
                r.Buf.Pinned = true;
                _inputs[v] = r;
                break;
        }

        _refs[v] = r;
        return r;
    }

    /// <summary>Like <see cref="Ref"/> but weights/constants are converted to dense f32.</summary>
    private TRef RefF32(Value v)
    {
        var r = Ref(v);
        if (!IsStatic(v) || r.Type == DataType.F32 || r.Type.Kind is DataTypeKind.Int or DataTypeKind.Bool)
        {
            return r;
        }

        if (_f32Cache.TryGetValue(v, out var cached))
        {
            return cached;
        }

        var format = Quant.FormatOf(r.Type) ?? throw new NotSupportedException($"Weight type {r.Type} cannot be converted to f32.");
        long elems = r.MaxElements;
        var buf = new Buf { Bytes = elems * 4, Ptr = AllocateOwned(elems * 4, false), Name = v.Name + ".f32" };
        Quant.Dequantize(format, r.Buf.Ptr, elems, (float*)buf.Ptr);
        cached = Register(new TRef { Type = DataType.F32, Dims = r.Dims, Strides = TRef.RowMajor(r.Dims), Buf = buf });
        _f32Cache[v] = cached;
        return cached;
    }
}
