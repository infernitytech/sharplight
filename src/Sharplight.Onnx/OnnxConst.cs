using System.Buffers.Binary;
using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

/// <summary>
/// A tensor whose value is known at import time (initializer, Constant node, folded shape computation).
/// Storage is lazy: byte-backed constants (mapped initializers) are only read when needed, and
/// pure data-movement ops (reshape, transpose) produce new constants without touching the bytes.
/// </summary>
internal sealed class OnnxConst
{
    public int ElemType { get; }
    public long[] Dims { get; }
    public long Count { get; }
    public string Name { get; init; } = "";

    private IWeightSource? _source;
    private byte[]? _bytes;
    private long[]? _ints;
    private double[]? _floats;
    private Func<byte[]>? _produce;

    private OnnxConst(int elemType, long[] dims)
    {
        ElemType = elemType;
        Dims = dims;
        long n = 1;
        foreach (var d in dims)
        {
            n *= d;
        }

        Count = n;
    }

    public int Rank => Dims.Length;
    public bool IsFloat => OnnxType.IsFloat(ElemType);
    public bool HasBackingSource => _source is not null && _bytes is null;
    public IWeightSource? Source => _source;
    public long ByteSize => OnnxType.ByteSize(ElemType, Count);

    /// <summary>Elements of a string constant (row-major); null for other types, whose data the string constant does not have.</summary>
    public string[]? Strings { get; private init; }

    public static OnnxConst FromTensor(OnnxTensor t)
        => t.Strings is { } strings ? FromStrings(t.Dims, strings, t.Name)
            : new(t.ElemType, t.Dims) { Name = t.Name, _source = t.Source ?? throw new InvalidDataException($"Tensor '{t.Name}' has no data.") };

    public static OnnxConst FromStrings(long[] dims, string[] strings, string name = "") => new(OnnxType.String, dims) { Name = name, Strings = strings };

    public static OnnxConst FromInts(int elemType, long[] dims, long[] values, string name = "")
    {
        var c = new OnnxConst(elemType, dims) { Name = name, _ints = values };
        if (values.Length != c.Count)
        {
            throw new ArgumentException($"{values.Length} values for shape [{string.Join(",", dims)}].");
        }

        return c;
    }

    public static OnnxConst FromFloats(int elemType, long[] dims, double[] values, string name = "")
    {
        var c = new OnnxConst(elemType, dims) { Name = name, _floats = values };
        if (values.Length != c.Count)
        {
            throw new ArgumentException($"{values.Length} values for shape [{string.Join(",", dims)}].");
        }

        return c;
    }

    public static OnnxConst FromBytes(int elemType, long[] dims, byte[] bytes, string name = "")
    {
        var c = new OnnxConst(elemType, dims) { Name = name, _bytes = bytes };
        if (bytes.Length != c.ByteSize)
        {
            throw new ArgumentException($"{bytes.Length} bytes for {OnnxType.Name(elemType)}[{string.Join(",", dims)}].");
        }

        return c;
    }

    public static OnnxConst Lazy(int elemType, long[] dims, Func<byte[]> produce, string name = "")
        => new(elemType, dims) { Name = name, _produce = produce };

    public static OnnxConst Scalar(float v) => FromFloats(OnnxType.Float, [], [v]);
    public static OnnxConst ScalarInt(long v) => FromInts(OnnxType.Int64, [], [v]);

    // ---- views (no data movement) ---------------------------------------------------------------
    /// <summary>Whether an integer element is a symbolic shape value (see <see cref="SymbolicInts"/>).</summary>
    public bool HasSymbols => _source is null && _produce is null && Count <= MaxSymbolicCount
        && (IsFloat ? Floats().Any(SymbolicInts.IsSymbolicDouble) : SymbolicInts.AnySymbolic(Ints()));

    // Symbolic elements only come from shape computations, which are small.
    private const long MaxSymbolicCount = 1 << 16;

    public OnnxConst Reshaped(long[] dims, string? name = null)
    {
        long n = 1;
        foreach (var d in dims)
        {
            n *= d;
        }

        if (n != Count)
        {
            throw new ArgumentException($"Cannot reshape {Count} elements to [{string.Join(",", dims)}].");
        }

        var c = new OnnxConst(ElemType, dims) { Name = name ?? Name, _source = _source, _bytes = _bytes, _ints = _ints, _floats = _floats };
        if (_produce is not null)
        {
            var me = this;
            c._produce = () => me.Bytes();
        }
        return c;
    }

    // ---- byte access ---------------------------------------------------------------------------
    public byte[] Bytes()
    {
        if (_bytes is not null)
        {
            return _bytes;
        }

        if (_produce is not null)
        {
            return _bytes = _produce();
        }

        if (_source is not null)
        {
            var b = new byte[_source.ByteLength];
            _source.Read(0, b);
            return _bytes = b;
        }
        return _bytes = Encode();
    }

    private byte[] Encode()
    {
        if (IsCoded(ElemType))
        {
            return EncodeCodes(OnnxType.NarrowFormat(ElemType)!.Value);
        }

        int es = OnnxType.ElementSize(ElemType);
        var b = new byte[Count * es];
        for (long i = 0; i < Count; i++)
        {
            var s = b.AsSpan((int)(i * es), es);
            if (_floats is not null)
            {
                double v = _floats[i];
                switch (ElemType)
                {
                    case OnnxType.Float:
                        BinaryPrimitives.WriteSingleLittleEndian(s, (float)v);
                        break;
                    case OnnxType.Double:
                        BinaryPrimitives.WriteDoubleLittleEndian(s, v);
                        break;
                    case OnnxType.Float16:
                        BinaryPrimitives.WriteHalfLittleEndian(s, (Half)v);
                        break;
                    case OnnxType.BFloat16:
                        BinaryPrimitives.WriteUInt16LittleEndian(s, (ushort)(BitConverter.SingleToUInt32Bits((float)v) >> 16));
                        break;
                    default:
                        WriteInt(s, es, (long)v);
                        break;
                }
            }
            else
            {
                WriteInt(s, es, _ints![i]);
            }
        }
        return b;
    }

    /// <summary>Bit patterns of the 8-bit and packed sub-byte types (low bits first within a byte).</summary>
    private byte[] EncodeCodes(NumericFormat format)
    {
        var codes = new byte[Count];
        for (long i = 0; i < Count; i++)
        {
            codes[i] = (byte)NumericFormats.Encode(format, _floats is not null ? _floats[i] : _ints![i]);
        }

        return PackCodes(codes, OnnxType.ElementBits(ElemType));
    }

    /// <summary>Value of element <paramref name="i"/> of an 8-bit or packed sub-byte type.</summary>
    private static double ReadCode(byte[] b, long i, int t)
    {
        int bits = OnnxType.ElementBits(t);
        long bit = i * bits;
        int code = (b[bit / 8] >> (int)(bit % 8)) & ((1 << bits) - 1);
        return NumericFormats.Decode(OnnxType.NarrowFormat(t)!.Value, code);
    }

    private byte[] UnpackCodes()
    {
        var b = Bytes();
        int bits = OnnxType.ElementBits(ElemType);
        var codes = new byte[Count];
        for (long i = 0; i < Count; i++)
        {
            long bit = i * bits;
            codes[i] = (byte)((b[bit / 8] >> (int)(bit % 8)) & ((1 << bits) - 1));
        }

        return codes;
    }

    private static byte[] PackCodes(byte[] codes, int bits)
    {
        var b = new byte[(codes.LongLength * bits + 7) / 8];
        for (long i = 0; i < codes.LongLength; i++)
        {
            long bit = i * bits;
            b[bit / 8] |= (byte)(codes[i] << (int)(bit % 8));
        }

        return b;
    }

    private static bool IsCoded(int t) => OnnxType.ElementBits(t) <= 8 && t is not (OnnxType.UInt8 or OnnxType.Int8 or OnnxType.Bool);

    private static void WriteInt(Span<byte> s, int es, long v)
    {
        switch (es)
        {
            case 1:
                s[0] = (byte)v;
                break;
            case 2:
                BinaryPrimitives.WriteInt16LittleEndian(s, (short)v);
                break;
            case 4:
                BinaryPrimitives.WriteInt32LittleEndian(s, (int)v);
                break;
            default:
                BinaryPrimitives.WriteInt64LittleEndian(s, v);
                break;
        }
    }

    // ---- host values -----------------------------------------------------------------------------
    public long[] Ints()
    {
        if (_ints is not null)
        {
            return _ints;
        }

        if (_floats is not null && _bytes is null && _source is null)
        {
            return _ints = _floats.Select(SymbolicInts.FromDouble).ToArray();
        }

        var b = Bytes();
        var r = new long[Count];
        if (IsCoded(ElemType))
        {
            for (long i = 0; i < Count; i++)
            {
                r[i] = (long)ReadCode(b, i, ElemType);
            }

            return _ints = r;
        }

        int es = OnnxType.ElementSize(ElemType);
        for (long i = 0; i < Count; i++)
        {
            var s = b.AsSpan((int)(i * es), es);
            r[i] = ElemType switch
            {
                OnnxType.Int64 or OnnxType.UInt64 => BinaryPrimitives.ReadInt64LittleEndian(s),
                OnnxType.Int32 => BinaryPrimitives.ReadInt32LittleEndian(s),
                OnnxType.UInt32 => BinaryPrimitives.ReadUInt32LittleEndian(s),
                OnnxType.Int16 => BinaryPrimitives.ReadInt16LittleEndian(s),
                OnnxType.UInt16 => BinaryPrimitives.ReadUInt16LittleEndian(s),
                OnnxType.Int8 => (sbyte)s[0],
                OnnxType.UInt8 or OnnxType.Bool => s[0],
                _ => (long)ReadFloat(s, ElemType),
            };
        }
        return _ints = r;
    }

    public double[] Floats()
    {
        if (_floats is not null)
        {
            return _floats;
        }

        if (_ints is not null && _bytes is null && _source is null)
        {
            return _floats = _ints.Select(SymbolicInts.ToDouble).ToArray();
        }

        var b = Bytes();
        var r = new double[Count];
        if (IsCoded(ElemType))
        {
            for (long i = 0; i < Count; i++)
            {
                r[i] = ReadCode(b, i, ElemType);
            }

            return _floats = r;
        }

        int es = OnnxType.ElementSize(ElemType);
        for (long i = 0; i < Count; i++)
        {
            var s = b.AsSpan((int)(i * es), es);
            r[i] = IsFloat ? ReadFloat(s, ElemType) : ElemType is OnnxType.Int64 or OnnxType.UInt64 ? BinaryPrimitives.ReadInt64LittleEndian(s) : ReadIntAsDouble(s, ElemType);
        }
        return _floats = r;
    }

    private static double ReadIntAsDouble(ReadOnlySpan<byte> s, int t) => t switch
    {
        OnnxType.Int32 => BinaryPrimitives.ReadInt32LittleEndian(s),
        OnnxType.UInt32 => BinaryPrimitives.ReadUInt32LittleEndian(s),
        OnnxType.Int16 => BinaryPrimitives.ReadInt16LittleEndian(s),
        OnnxType.UInt16 => BinaryPrimitives.ReadUInt16LittleEndian(s),
        OnnxType.Int8 => (sbyte)s[0],
        _ => s[0],
    };

    private static double ReadFloat(ReadOnlySpan<byte> s, int t) => t switch
    {
        OnnxType.Float => BinaryPrimitives.ReadSingleLittleEndian(s),
        OnnxType.Double => BinaryPrimitives.ReadDoubleLittleEndian(s),
        OnnxType.Float16 => (double)BinaryPrimitives.ReadHalfLittleEndian(s),
        OnnxType.BFloat16 => BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(s) << 16),
        _ => throw new NotSupportedException($"Not a float type: {OnnxType.Name(t)}"),
    };

    /// <summary>Little-endian f32 bytes of the (float) data, converting from f16/bf16/f64/ints as needed.</summary>
    public byte[] F32Bytes()
    {
        if (ElemType == OnnxType.Float)
        {
            return Bytes();
        }

        var f = Floats();
        var r = new byte[Count * 4];
        for (long i = 0; i < Count; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(r.AsSpan((int)(i * 4), 4), (float)f[i]);
        }

        return r;
    }

    public float[] F32()
    {
        var b = F32Bytes();
        var r = new float[Count];
        MemoryMarshal.Cast<byte, float>(b).CopyTo(r);
        return r;
    }

    public float ScalarF32() => Count == 1 ? (float)Floats()[0] : throw new InvalidOperationException($"Expected a scalar, got {Count} elements.");
    public long ScalarInt() => Count == 1 ? Ints()[0] : throw new InvalidOperationException($"Expected a scalar, got {Count} elements.");

    // ---- element-type conversion -----------------------------------------------------------------
    /// <summary>
    /// Converts to element type <paramref name="et"/> with ONNX Cast semantics: floats round to nearest even (FP8 saturating when
    /// <paramref name="saturate"/>), floats truncate toward zero to integers, integers wrap to narrower integer types. Symbolic shape
    /// values pass through unchanged.
    /// </summary>
    public OnnxConst CastTo(int et, bool saturate = true)
    {
        if (et == ElemType)
        {
            return this;
        }

        var format = OnnxType.NarrowFormat(et);
        if (OnnxType.IsFloat(et))
        {
            var f = Floats();
            if (et == OnnxType.Float)
            {
                f = f.Select(v => SymbolicInts.IsSymbolicDouble(v) ? v : (float)v).ToArray();
            }
            else if (format is { } narrow)
            {
                f = f.Select(v => SymbolicInts.IsSymbolicDouble(v) ? v : NumericFormats.Round(narrow, (float)v, saturate)).ToArray();
            }

            return FromFloats(et, Dims, f, Name);
        }
        if (et == OnnxType.Bool)
        {
            return FromInts(et, Dims, (IsFloat ? Floats().Select(v => v != 0 ? 1L : 0L) : Ints().Select(v => v != 0 ? 1L : 0L)).ToArray(), Name);
        }

        var ints = IsFloat ? Floats().Select(SymbolicInts.FromDouble).ToArray() : Ints();
        if (format is { } integer && !SymbolicInts.AnySymbolic(ints))
        {
            var (min, _) = NumericFormats.IntegerRange(integer);
            ints = ints.Select(v => NumericFormats.Wrap(v, NumericFormats.Bits(integer), min < 0)).ToArray();
        }

        return FromInts(et, Dims, ints, Name);
    }

    // ---- data movement --------------------------------------------------------------------------
    public OnnxConst Transposed(int[] perm)
    {
        var src = this;
        var newDims = perm.Select(p => Dims[p]).ToArray();
        if (OnnxType.IsSubByte(ElemType))
        {
            // Move one code per byte, then pack again.
            var codes = FromBytes(OnnxType.UInt8, Dims, UnpackCodes()).Transposed(perm);
            int bits = OnnxType.ElementBits(ElemType);
            return Lazy(ElemType, newDims, () => PackCodes(codes.Bytes(), bits), Name);
        }

        int es = OnnxType.ElementSize(ElemType);
        return Lazy(ElemType, newDims, () =>
        {
            var inB = src.Bytes();
            var outB = new byte[inB.Length];
            int r = perm.Length;
            var inStr = new long[r];
            long s = 1;
            for (int d = r - 1; d >= 0; d--)
            {
                inStr[d] = s;
                s *= src.Dims[d];
            }
            var pStr = perm.Select(p => inStr[p]).ToArray();
            long total = src.Count;
            if (total == 0)
            {
                return outB;
            }
            // iterate over output indices in row-major order
            var idx = new long[r];
            long inPos = 0;
            for (long o = 0; o < total; o++)
            {
                Buffer.BlockCopy(inB, (int)(inPos * es), outB, (int)(o * es), es);
                for (int d = r - 1; d >= 0; d--)
                {
                    idx[d]++;
                    inPos += pStr[d];
                    if (idx[d] < newDims[d])
                    {
                        break;
                    }

                    inPos -= pStr[d] * newDims[d];
                    idx[d] = 0;
                }
            }
            return outB;
        }, Name);
    }

    public override string ToString() => $"const {OnnxType.Name(ElemType)}[{string.Join(",", Dims)}] {Name}";
}
