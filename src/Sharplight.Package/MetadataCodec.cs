using System.Text;
using Sharplight.Tensors;

namespace Sharplight.Package;

/// <summary>Typed values of metadata, properties and op attributes (scalars, strings, arrays, data types).</summary>
internal static class MetadataCodec
{
    private enum Tag : byte { String = 1, Bool, I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, DataType, Array = 0x40 }

    public static void WriteString(BinaryWriter w, string s)
    {
        var b = Encoding.UTF8.GetBytes(s);
        w.Write(b.Length);
        w.Write(b);
    }

    public static string ReadString(BinaryReader r) => Encoding.UTF8.GetString(r.ReadBytes(r.ReadInt32()));

    public static void Write(BinaryWriter w, IReadOnlyDictionary<string, object> values)
    {
        var ordered = values.OrderBy(kv => kv.Key, StringComparer.Ordinal).ToList();
        w.Write(ordered.Count);
        foreach (var (k, v) in ordered)
        {
            WriteString(w, k);
            try { WriteValue(w, v); }
            catch (NotSupportedException e) { throw new NotSupportedException($"'{k}': {e.Message}", e); }
        }
    }

    public static Dictionary<string, object> Read(BinaryReader r)
    {
        int n = r.ReadInt32();
        var d = new Dictionary<string, object>(n, StringComparer.Ordinal);
        for (int i = 0; i < n; i++)
        {
            d[ReadString(r)] = ReadValue(r);
        }

        return d;
    }

    public static void WriteValue(BinaryWriter w, object v)
    {
        if (v is Array a)
        {
            var et = a.GetType().GetElementType()!;
            if (et == typeof(object))
            {
                // Heterogeneous holder: use the element type of its contents (must agree).
                var types = a.Cast<object>().Select(o => o.GetType()).Distinct().ToList();
                if (types.Count > 1)
                {
                    throw new NotSupportedException($"mixed array element types ({string.Join(", ", types.Select(t => t.Name))})");
                }

                et = types.Count == 1 ? types[0] : typeof(long);
            }
            var tag = TagOf(et);
            w.Write((byte)(Tag.Array | tag));
            w.Write(a.LongLength);
            foreach (var e in a)
            {
                WriteScalar(w, tag, e);
            }

            return;
        }
        var t = TagOf(v.GetType());
        w.Write((byte)t);
        WriteScalar(w, t, v);
    }

    public static object ReadValue(BinaryReader r)
    {
        var tag = (Tag)r.ReadByte();
        if ((tag & Tag.Array) == 0)
        {
            return ReadScalar(r, tag);
        }

        var et = tag & ~Tag.Array;
        long n = r.ReadInt64();
        var arr = Array.CreateInstance(ClrType(et), n);
        for (long i = 0; i < n; i++)
        {
            arr.SetValue(ReadScalar(r, et), i);
        }

        return arr;
    }

    private static Tag TagOf(Type t) => t switch
    {
        _ when t == typeof(string) => Tag.String,
        _ when t == typeof(bool) => Tag.Bool,
        _ when t == typeof(sbyte) => Tag.I8,
        _ when t == typeof(byte) => Tag.U8,
        _ when t == typeof(short) => Tag.I16,
        _ when t == typeof(ushort) => Tag.U16,
        _ when t == typeof(int) => Tag.I32,
        _ when t == typeof(uint) => Tag.U32,
        _ when t == typeof(long) => Tag.I64,
        _ when t == typeof(ulong) => Tag.U64,
        _ when t == typeof(float) => Tag.F32,
        _ when t == typeof(double) => Tag.F64,
        _ when t == typeof(DataType) => Tag.DataType,
        _ => throw new NotSupportedException($"value type {t.FullName} cannot be stored in a package"),
    };

    private static Type ClrType(Tag t) => t switch
    {
        Tag.String => typeof(string), Tag.Bool => typeof(bool), Tag.I8 => typeof(sbyte), Tag.U8 => typeof(byte),
        Tag.I16 => typeof(short), Tag.U16 => typeof(ushort), Tag.I32 => typeof(int), Tag.U32 => typeof(uint),
        Tag.I64 => typeof(long), Tag.U64 => typeof(ulong), Tag.F32 => typeof(float), Tag.F64 => typeof(double),
        Tag.DataType => typeof(DataType),
        _ => throw new InvalidDataException($"Unknown package value tag {t}."),
    };

    private static void WriteScalar(BinaryWriter w, Tag t, object v)
    {
        switch (t)
        {
            case Tag.String: WriteString(w, (string)v); break;
            case Tag.Bool: w.Write((bool)v); break;
            case Tag.I8: w.Write((sbyte)v); break;
            case Tag.U8: w.Write((byte)v); break;
            case Tag.I16: w.Write((short)v); break;
            case Tag.U16: w.Write((ushort)v); break;
            case Tag.I32: w.Write((int)v); break;
            case Tag.U32: w.Write((uint)v); break;
            case Tag.I64: w.Write((long)v); break;
            case Tag.U64: w.Write((ulong)v); break;
            case Tag.F32: w.Write((float)v); break;
            case Tag.F64: w.Write((double)v); break;
            case Tag.DataType: WriteString(w, ((DataType)v).Name); break;
            default: throw new NotSupportedException(t.ToString());
        }
    }

    private static object ReadScalar(BinaryReader r, Tag t) => t switch
    {
        Tag.String => ReadString(r),
        Tag.Bool => r.ReadBoolean(),
        Tag.I8 => r.ReadSByte(),
        Tag.U8 => r.ReadByte(),
        Tag.I16 => r.ReadInt16(),
        Tag.U16 => r.ReadUInt16(),
        Tag.I32 => r.ReadInt32(),
        Tag.U32 => r.ReadUInt32(),
        Tag.I64 => r.ReadInt64(),
        Tag.U64 => r.ReadUInt64(),
        Tag.F32 => r.ReadSingle(),
        Tag.F64 => r.ReadDouble(),
        Tag.DataType => DataType.Get(ReadString(r)),
        _ => throw new InvalidDataException($"Unknown package value tag {t}."),
    };

    public static void WriteShape(BinaryWriter w, Shape s)
    {
        w.Write(s.Rank);
        foreach (var d in s.Dims)
        {
            w.Write(d.IsSymbolic);
            if (d.IsSymbolic)
            {
                WriteString(w, d.Symbol!);
            }
            else
            {
                w.Write(d.Fixed);
            }
        }
    }

    public static Shape ReadShape(BinaryReader r)
    {
        int rank = r.ReadInt32();
        var dims = new Dim[rank];
        for (int i = 0; i < rank; i++)
        {
            dims[i] = r.ReadBoolean() ? Dim.Sym(ReadString(r)) : Dim.Of(r.ReadInt64());
        }

        return new Shape(dims);
    }
}
