using System.Text;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>Minimal GGUF v3 writer, mainly used to create test fixtures.</summary>
public sealed class GgufWriter
{
    readonly List<(string key, object value)> _kv = [];
    readonly List<(string name, int type, long[] ne, ReadOnlyMemory<byte> data)> _tensors = [];

    public int Alignment { get; set; } = GgufFile.DefaultAlignment;

    /// <summary>Adds a KV. CLR type selects the GGUF type; T[] writes an array, object[] a nested array.</summary>
    public GgufWriter AddMetadata(string key, object value) { _kv.Add((key, value)); return this; }

    /// <summary>Adds a tensor with ggml-order dimensions (<paramref name="ne"/> innermost first).</summary>
    public GgufWriter AddTensorNe(string name, DataType type, ReadOnlyMemory<byte> data, params long[] ne)
    {
        long expected = GgufFile.ComputeByteSize(type, ne, name);
        if (data.Length != expected) throw new ArgumentException($"Tensor '{name}': expected {expected} bytes, got {data.Length}.");
        _tensors.Add((name, GgmlTypes.GetGgmlType(type), ne, data));
        return this;
    }

    /// <summary>Adds a tensor given its row-major (outermost-first) dimensions.</summary>
    public GgufWriter AddTensor(string name, DataType type, ReadOnlyMemory<byte> data, params long[] dims)
        => AddTensorNe(name, type, data, dims.Reverse().ToArray());

    public void Write(string path)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16);
        Write(fs);
    }

    public void Write(Stream stream)
    {
        using var w = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        var kv = _kv.ToList();
        if (Alignment != GgufFile.DefaultAlignment && !kv.Any(k => k.key == "general.alignment"))
            kv.Add(("general.alignment", (uint)Alignment));

        w.Write(GgufFile.Magic);
        w.Write(3u);
        w.Write((ulong)_tensors.Count);
        w.Write((ulong)kv.Count);
        foreach (var (k, v) in kv) { WriteString(w, k); WriteTyped(w, v); }

        long off = 0;
        var offsets = new long[_tensors.Count];
        for (int i = 0; i < _tensors.Count; i++)
        {
            var t = _tensors[i];
            offsets[i] = off;
            WriteString(w, t.name);
            w.Write((uint)t.ne.Length);
            foreach (var n in t.ne) w.Write((ulong)n);
            w.Write((uint)t.type);
            w.Write((ulong)off);
            off += Align(t.data.Length);
        }
        Pad(w, stream.Position);
        for (int i = 0; i < _tensors.Count; i++)
        {
            w.Write(_tensors[i].data.Span);
            Pad(w, stream.Position);
        }
        w.Flush();
    }

    long Align(long n) => (n + Alignment - 1) / Alignment * Alignment;

    void Pad(BinaryWriter w, long pos)
    {
        long pad = Align(pos) - pos;
        if (pad > 0) w.Write(new byte[pad]);
    }

    static void WriteString(BinaryWriter w, string s)
    {
        var b = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)b.Length);
        w.Write(b);
    }

    static GgufValueType TypeOf(Type t)
    {
        if (t == typeof(byte)) return GgufValueType.UInt8;
        if (t == typeof(sbyte)) return GgufValueType.Int8;
        if (t == typeof(ushort)) return GgufValueType.UInt16;
        if (t == typeof(short)) return GgufValueType.Int16;
        if (t == typeof(uint)) return GgufValueType.UInt32;
        if (t == typeof(int)) return GgufValueType.Int32;
        if (t == typeof(float)) return GgufValueType.Float32;
        if (t == typeof(bool)) return GgufValueType.Bool;
        if (t == typeof(string)) return GgufValueType.String;
        if (t == typeof(ulong)) return GgufValueType.UInt64;
        if (t == typeof(long)) return GgufValueType.Int64;
        if (t == typeof(double)) return GgufValueType.Float64;
        if (t == typeof(object)) return GgufValueType.Array; // object[] elements are nested arrays
        if (t.IsArray) return GgufValueType.Array;
        throw new NotSupportedException($"Cannot write metadata of type {t}.");
    }

    static void WriteTyped(BinaryWriter w, object v)
    {
        var t = v is Array ? GgufValueType.Array : TypeOf(v.GetType());
        w.Write((uint)t);
        WriteValue(w, v);
    }

    static void WriteValue(BinaryWriter w, object v)
    {
        switch (v)
        {
            case byte x: w.Write(x); break;
            case sbyte x: w.Write(x); break;
            case ushort x: w.Write(x); break;
            case short x: w.Write(x); break;
            case uint x: w.Write(x); break;
            case int x: w.Write(x); break;
            case float x: w.Write(x); break;
            case bool x: w.Write((byte)(x ? 1 : 0)); break;
            case string x: WriteString(w, x); break;
            case ulong x: w.Write(x); break;
            case long x: w.Write(x); break;
            case double x: w.Write(x); break;
            case Array a:
                w.Write((uint)TypeOf(a.GetType().GetElementType()!));
                w.Write((ulong)a.Length);
                foreach (var e in a) WriteValue(w, e);
                break;
            default: throw new NotSupportedException($"Cannot write metadata of type {v.GetType()}.");
        }
    }
}
