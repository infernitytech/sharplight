using System.Text;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>Minimal GGUF v3 writer, used to create test fixtures.</summary>
internal sealed class GgufWriter
{
    private readonly List<(string key, object value)> _kv = [];
    private readonly List<(string name, int type, long[] ne, ReadOnlyMemory<byte> data)> _tensors = [];

    /// <summary>Tensor data alignment in bytes; written as <c>general.alignment</c> when it differs from the default.</summary>
    public int Alignment { get; set; } = GgufFile.DefaultAlignment;

    /// <summary>Adds a KV. CLR type selects the GGUF type; T[] writes an array, object[] a nested array.</summary>
    public GgufWriter AddMetadata(string key, object value)
    {
        _kv.Add((key, value));
        return this;
    }

    /// <summary>Adds a tensor with ggml-order dimensions (<paramref name="ne"/> innermost first).</summary>
    public GgufWriter AddTensorNe(string name, DataType type, ReadOnlyMemory<byte> data, params long[] ne)
    {
        long expected = GgufFile.ComputeByteSize(type, ne, name);
        if (data.Length != expected)
        {
            throw new ArgumentException($"Tensor '{name}': expected {expected} bytes, got {data.Length}.");
        }

        _tensors.Add((name, GgmlTypes.GetGgmlType(type), ne, data));
        return this;
    }

    /// <summary>Adds a tensor given its row-major (outermost-first) dimensions.</summary>
    public GgufWriter AddTensor(string name, DataType type, ReadOnlyMemory<byte> data, params long[] dims)
        => AddTensorNe(name, type, data, dims.Reverse().ToArray());

    /// <summary>Writes the file to <paramref name="path"/>, replacing an existing file.</summary>
    public void Write(string path)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16);
        Write(fs);
    }

    /// <summary>Writes the file to <paramref name="stream"/>, which stays open.</summary>
    public void Write(Stream stream)
    {
        using var w = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        var kv = _kv.ToList();
        if (Alignment != GgufFile.DefaultAlignment && !kv.Any(k => k.key == "general.alignment"))
        {
            kv.Add(("general.alignment", (uint)Alignment));
        }

        w.Write(GgufFile.Magic);
        w.Write(3u);
        w.Write((ulong)_tensors.Count);
        w.Write((ulong)kv.Count);
        foreach (var (key, value) in kv)
        {
            WriteString(w, key);
            WriteTyped(w, value);
        }

        long off = 0;
        for (int i = 0; i < _tensors.Count; i++)
        {
            var t = _tensors[i];
            WriteString(w, t.name);
            w.Write((uint)t.ne.Length);
            foreach (var n in t.ne)
            {
                w.Write((ulong)n);
            }

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

    private long Align(long n) => (n + Alignment - 1) / Alignment * Alignment;

    private void Pad(BinaryWriter w, long pos)
    {
        long pad = Align(pos) - pos;
        if (pad > 0)
        {
            w.Write(new byte[pad]);
        }
    }

    private static void WriteString(BinaryWriter w, string s)
    {
        var b = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)b.Length);
        w.Write(b);
    }

    private static readonly Dictionary<Type, GgufValueType> s_valueTypes = new()
    {
        [typeof(byte)] = GgufValueType.UInt8,
        [typeof(sbyte)] = GgufValueType.Int8,
        [typeof(ushort)] = GgufValueType.UInt16,
        [typeof(short)] = GgufValueType.Int16,
        [typeof(uint)] = GgufValueType.UInt32,
        [typeof(int)] = GgufValueType.Int32,
        [typeof(float)] = GgufValueType.Float32,
        [typeof(bool)] = GgufValueType.Bool,
        [typeof(string)] = GgufValueType.String,
        [typeof(ulong)] = GgufValueType.UInt64,
        [typeof(long)] = GgufValueType.Int64,
        [typeof(double)] = GgufValueType.Float64,
        [typeof(object)] = GgufValueType.Array,   // object[] elements are nested arrays
    };

    private static GgufValueType TypeOf(Type t)
    {
        if (s_valueTypes.TryGetValue(t, out var valueType))
        {
            return valueType;
        }

        if (t.IsArray)
        {
            return GgufValueType.Array;
        }

        throw new NotSupportedException($"Cannot write metadata of type {t}.");
    }

    private static void WriteTyped(BinaryWriter w, object v)
    {
        var t = v is Array ? GgufValueType.Array : TypeOf(v.GetType());
        w.Write((uint)t);
        WriteValue(w, v);
    }

    private static void WriteValue(BinaryWriter w, object v)
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
                foreach (var e in a)
                {
                    WriteValue(w, e);
                }

                break;
            default: throw new NotSupportedException($"Cannot write metadata of type {v.GetType()}.");
        }
    }
}
