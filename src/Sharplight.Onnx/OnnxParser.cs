using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using Sharplight.Models;

namespace Sharplight.Onnx;

/// <summary>Creates weight sources for byte ranges of the file being parsed.</summary>
public interface IOnnxByteSliceProvider
{
    public IWeightSource Slice(long offset, long length);
}

/// <summary>Resolves <c>data_location = EXTERNAL</c> tensor data (usually a <c>.onnx.data</c> file next to the model).</summary>
public interface IOnnxExternalDataResolver
{
    /// <param name="length">Byte length, or -1 for "to the end of the file".</param>
    public IWeightSource Open(string location, long offset, long length);
}

/// <summary>A read-only memory-mapped file whose address stays valid until disposed.</summary>
public sealed unsafe class MappedFile : IDisposable
{
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _view;
    private byte* _base;
    public string Path { get; }
    public long Length { get; }
    public byte* Pointer => _base;

    public MappedFile(string path)
    {
        Path = path;
        var fi = new FileInfo(path);
        Length = fi.Length;
        if (Length == 0)
        {
            throw new InvalidDataException($"{path} is empty.");
        }

        _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _view = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref _base);
        _base += _view.PointerOffset;
    }

    public ReadOnlySpan<byte> AsSpan(long offset = 0, long length = -1)
    {
        if (length < 0)
        {
            length = Length - offset;
        }

        return new ReadOnlySpan<byte>(_base + offset, checked((int)length));
    }

    public void Dispose()
    {
        if (_base != null) { _view.SafeMemoryMappedViewHandle.ReleasePointer(); _base = null; }
        _view.Dispose();
        _mmf.Dispose();
    }
}

/// <summary>A byte range of a <see cref="MappedFile"/> exposed as a zero-copy weight source.</summary>
public sealed unsafe class MappedRegionSource(MappedFile file, long offset, long length) : IWeightSource
{
    public long ByteLength => length;
    public void Read(long off, Span<byte> destination)
    {
        if (off < 0 || off + destination.Length > length)
        {
            throw new ArgumentOutOfRangeException(nameof(off));
        }

        new ReadOnlySpan<byte>(file.Pointer + offset + off, destination.Length).CopyTo(destination);
    }
    public nint TryGetPointer() => (nint)(file.Pointer + offset);
}

/// <summary>An in-memory weight source (typed tensor data, tests).</summary>
public sealed class ArrayWeightSource(byte[] data, int offset = 0, int length = -1) : IWeightSource
{
    private readonly int _length = length < 0 ? data.Length - offset : length;
    public long ByteLength => _length;
    public void Read(long off, Span<byte> destination)
    {
        if (off < 0 || off + destination.Length > _length)
        {
            throw new ArgumentOutOfRangeException(nameof(off));
        }

        data.AsSpan(offset + (int)off, destination.Length).CopyTo(destination);
    }
}

internal sealed class ArraySliceProvider(byte[] data) : IOnnxByteSliceProvider
{
    public IWeightSource Slice(long offset, long length) => new ArrayWeightSource(data, (int)offset, (int)length);
}

internal sealed class MappedSliceProvider(MappedFile file) : IOnnxByteSliceProvider
{
    public IWeightSource Slice(long offset, long length) => new MappedRegionSource(file, offset, length);
}

/// <summary>Decodes ModelProto (the subset of onnx.proto Sharplight needs).</summary>
public static class OnnxParser
{
    /// <summary>Parses a model held in memory (tensor data is copied lazily from <paramref name="data"/>).</summary>
    public static OnnxModel Parse(byte[] data, IOnnxExternalDataResolver? external = null)
        => Parse(data, new ArraySliceProvider(data), external);

    public static OnnxModel Parse(ReadOnlySpan<byte> data, IOnnxByteSliceProvider slices, IOnnxExternalDataResolver? external = null)
    {
        var ctx = new Ctx(data.Length, slices, external);
        var r = new ProtoReader(data);
        long irVersion = 0, modelVersion = 0;
        string producer = "", producerVersion = "", domain = "", doc = "";
        var opsets = new Dictionary<string, long>();
        var meta = new Dictionary<string, string>();
        OnnxGraph? graph = null;
        while (r.Next(out int f, out var w))
        {
            switch (f)
            {
                case 1 when w == WireType.Varint: irVersion = r.ReadInt64(); break;
                case 2 when w == WireType.LengthDelimited: producer = r.ReadString(); break;
                case 3 when w == WireType.LengthDelimited: producerVersion = r.ReadString(); break;
                case 4 when w == WireType.LengthDelimited: domain = r.ReadString(); break;
                case 5 when w == WireType.Varint: modelVersion = r.ReadInt64(); break;
                case 6 when w == WireType.LengthDelimited: doc = r.ReadString(); break;
                case 7 when w == WireType.LengthDelimited: graph = ParseGraph(r.ReadMessage(), ctx, data); break;
                case 8 when w == WireType.LengthDelimited:
                {
                    var m = r.ReadMessage(); string d = ""; long v = 0;
                    while (m.Next(out int mf, out var mw))
                    {
                        if (mf == 1 && mw == WireType.LengthDelimited)
                            {
                                d = m.ReadString();
                            }
                            else if (mf == 2 && mw == WireType.Varint)
                            {
                                v = m.ReadInt64();
                            }
                            else
                            {
                                m.Skip(mw);
                            }
                        }
                    opsets[d] = v;
                    break;
                }
                case 14 when w == WireType.LengthDelimited:
                {
                    var (k, v) = ParseStringEntry(r.ReadMessage());
                    meta[k] = v;
                    break;
                }
                default: r.Skip(w); break;
            }
        }
        return new OnnxModel
        {
            IrVersion = irVersion, ProducerName = producer, ProducerVersion = producerVersion, Domain = domain,
            ModelVersion = modelVersion, DocString = doc, Opsets = opsets, MetadataProps = meta,
            Graph = graph ?? throw new InvalidDataException("ONNX model has no graph."),
        };
    }

    private sealed record Ctx(long FileLength, IOnnxByteSliceProvider Slices, IOnnxExternalDataResolver? External);

    private static (string, string) ParseStringEntry(ProtoReader m)
    {
        string k = "", v = "";
        while (m.Next(out int f, out var w))
        {
            if (f == 1 && w == WireType.LengthDelimited)
            {
                k = m.ReadString();
            }
            else if (f == 2 && w == WireType.LengthDelimited)
            {
                v = m.ReadString();
            }
            else
            {
                m.Skip(w);
            }
        }
        return (k, v);
    }

    private static OnnxGraph ParseGraph(ProtoReader r, Ctx ctx, ReadOnlySpan<byte> data)
    {
        string name = "", doc = "";
        var nodes = new List<OnnxNode>();
        var inits = new List<OnnxTensor>();
        var inputs = new List<OnnxValueInfo>();
        var outputs = new List<OnnxValueInfo>();
        var vi = new List<OnnxValueInfo>();
        while (r.Next(out int f, out var w))
        {
            if (w != WireType.LengthDelimited) { r.Skip(w); continue; }
            switch (f)
            {
                case 1: nodes.Add(ParseNode(r.ReadMessage(), ctx, data)); break;
                case 2: name = r.ReadString(); break;
                case 5: inits.Add(ParseTensor(r.ReadMessage(), ctx, data)); break;
                case 10: doc = r.ReadString(); break;
                case 11: inputs.Add(ParseValueInfo(r.ReadMessage())); break;
                case 12: outputs.Add(ParseValueInfo(r.ReadMessage())); break;
                case 13: vi.Add(ParseValueInfo(r.ReadMessage())); break;
                default: r.Skip(w); break;
            }
        }
        return new OnnxGraph { Name = name, DocString = doc, Nodes = nodes, Initializers = inits, Inputs = inputs, Outputs = outputs, ValueInfo = vi };
    }

    private static OnnxNode ParseNode(ProtoReader r, Ctx ctx, ReadOnlySpan<byte> data)
    {
        string name = "", op = "", domain = "";
        var ins = new List<string>(); var outs = new List<string>();
        var attrs = new Dictionary<string, OnnxAttribute>(StringComparer.Ordinal);
        while (r.Next(out int f, out var w))
        {
            if (w != WireType.LengthDelimited) { r.Skip(w); continue; }
            switch (f)
            {
                case 1: ins.Add(r.ReadString()); break;
                case 2: outs.Add(r.ReadString()); break;
                case 3: name = r.ReadString(); break;
                case 4: op = r.ReadString(); break;
                case 5: { var a = ParseAttribute(r.ReadMessage(), ctx, data); attrs[a.Name] = a; break; }
                case 7: domain = r.ReadString(); break;
                default: r.Skip(w); break;
            }
        }
        return new OnnxNode { Name = name, OpType = op, Domain = domain, Inputs = ins, Outputs = outs, Attributes = attrs };
    }

    private static OnnxAttribute ParseAttribute(ProtoReader r, Ctx ctx, ReadOnlySpan<byte> data)
    {
        string name = ""; int type = 0; float fv = 0; long iv = 0; string? sv = null; OnnxTensor? t = null; OnnxGraph? g = null;
        var floats = new List<float>(); var ints = new List<long>(); var strs = new List<string>();
        bool hasF = false, hasI = false, hasS = false, hasT = false;
        while (r.Next(out int f, out var w))
        {
            switch (f)
            {
                case 1 when w == WireType.LengthDelimited: name = r.ReadString(); break;
                case 2 when w == WireType.Fixed32: fv = r.ReadFloat(); hasF = true; break;
                case 3 when w == WireType.Varint: iv = r.ReadInt64(); hasI = true; break;
                case 4 when w == WireType.LengthDelimited: sv = System.Text.Encoding.UTF8.GetString(r.ReadBytes(out _)); hasS = true; break;
                case 5 when w == WireType.LengthDelimited: t = ParseTensor(r.ReadMessage(), ctx, data); hasT = true; break;
                case 6 when w == WireType.LengthDelimited: g = ParseGraph(r.ReadMessage(), ctx, data); break;
                case 7: r.ReadRepeatedFloat(w, floats); break;
                case 8: r.ReadRepeatedInt64(w, ints); break;
                case 9 when w == WireType.LengthDelimited: strs.Add(System.Text.Encoding.UTF8.GetString(r.ReadBytes(out _))); break;
                case 20 when w == WireType.Varint: type = (int)r.ReadInt64(); break;
                default: r.Skip(w); break;
            }
        }
        if (type == 0)
        {
            type = hasF ? 1 : hasI ? 2 : hasS ? 3 : hasT ? 4 : g is not null ? 5 : floats.Count > 0 ? 6 : ints.Count > 0 ? 7 : strs.Count > 0 ? 8 : 0;
        }

        return new OnnxAttribute { Name = name, Type = (OnnxAttrType)type, F = fv, I = iv, S = sv, T = t, G = g, Floats = floats, Ints = ints, Strings = strs };
    }

    private static OnnxValueInfo ParseValueInfo(ProtoReader r)
    {
        string name = ""; int elem = 0; List<OnnxDim>? shape = null;
        while (r.Next(out int f, out var w))
        {
            if (f == 1 && w == WireType.LengthDelimited)
            {
                name = r.ReadString();
            }
            else if (f == 2 && w == WireType.LengthDelimited)
            {
                var type = r.ReadMessage();
                while (type.Next(out int tf, out var tw))
                {
                    if (tf == 1 && tw == WireType.LengthDelimited) // tensor_type
                    {
                        var tt = type.ReadMessage();
                        while (tt.Next(out int ef, out var ew))
                        {
                            if (ef == 1 && ew == WireType.Varint)
                            {
                                elem = (int)tt.ReadInt64();
                            }
                            else if (ef == 2 && ew == WireType.LengthDelimited)
                            {
                                shape = [];
                                var sh = tt.ReadMessage();
                                while (sh.Next(out int sf, out var sw))
                                {
                                    if (sf != 1 || sw != WireType.LengthDelimited) { sh.Skip(sw); continue; }
                                    var dm = sh.ReadMessage();
                                    long? val = null; string? param = null;
                                    while (dm.Next(out int df, out var dw))
                                    {
                                        if (df == 1 && dw == WireType.Varint)
                                        {
                                            val = dm.ReadInt64();
                                        }
                                        else if (df == 2 && dw == WireType.LengthDelimited)
                                        {
                                            param = dm.ReadString();
                                        }
                                        else
                                        {
                                            dm.Skip(dw);
                                        }
                                    }
                                    shape.Add(new OnnxDim(val, string.IsNullOrEmpty(param) ? null : param));
                                }
                            }
                            else
                            {
                                tt.Skip(ew);
                            }
                        }
                    }
                    else
                    {
                        type.Skip(tw);
                    }
                }
            }
            else
            {
                r.Skip(w);
            }
        }
        return new OnnxValueInfo { Name = name, ElemType = elem, Shape = shape };
    }

    private static OnnxTensor ParseTensor(ProtoReader r, Ctx ctx, ReadOnlySpan<byte> data)
    {
        var dims = new List<long>();
        int dtype = 0; string name = "";
        int rawOff = -1, rawLen = 0;
        var floatData = new List<float>(); var int32Data = new List<long>(); var int64Data = new List<long>();
        var doubleData = new List<double>(); var uint64Data = new List<long>();
        var ext = new Dictionary<string, string>();
        long dataLocation = 0;
        while (r.Next(out int f, out var w))
        {
            switch (f)
            {
                case 1: r.ReadRepeatedInt64(w, dims); break;
                case 2 when w == WireType.Varint: dtype = (int)r.ReadInt64(); break;
                case 4: r.ReadRepeatedFloat(w, floatData); break;
                case 5: r.ReadRepeatedInt64(w, int32Data); break;
                case 7: r.ReadRepeatedInt64(w, int64Data); break;
                case 8 when w == WireType.LengthDelimited: name = r.ReadString(); break;
                case 9 when w == WireType.LengthDelimited: { var b = r.ReadBytes(out rawOff); rawLen = b.Length; break; }
                case 10: r.ReadRepeatedDouble(w, doubleData); break;
                case 11: r.ReadRepeatedInt64(w, uint64Data); break;
                case 13 when w == WireType.LengthDelimited: { var (k, v) = ParseStringEntry(r.ReadMessage()); ext[k] = v; break; }
                case 14 when w == WireType.Varint: dataLocation = r.ReadInt64(); break;
                default: r.Skip(w); break;
            }
        }
        var dimsArr = dims.ToArray();
        long elems = 1;
        foreach (var d in dimsArr) { if (d < 0) { throw new InvalidDataException($"Tensor '{name}' has a negative dimension."); } elems = checked(elems * d); }

        IWeightSource? src = null;
        if (dtype == OnnxType.String) { }
        else if (dataLocation == 1)
        {
            if (!ext.TryGetValue("location", out var loc))
            {
                throw new InvalidDataException($"External tensor '{name}' has no location.");
            }

            long off = ext.TryGetValue("offset", out var os) ? long.Parse(os) : 0;
            long len = ext.TryGetValue("length", out var ls) ? long.Parse(ls) : OnnxType.ElementSize(dtype) * elems;
            if (ctx.External is null)
            {
                throw new NotSupportedException($"Tensor '{name}' uses external data '{loc}' but no resolver is available.");
            }

            src = ctx.External.Open(loc, off, len);
        }
        else if (rawOff >= 0)
        {
            src = ctx.Slices.Slice(rawOff, rawLen);
        }
        else if (dtype == OnnxType.Float && floatData.Count > 0)
        {
            src = Bytes(floatData.Count * 4, s => { for (int i = 0; i < floatData.Count; i++) { BinaryPrimitives.WriteSingleLittleEndian(s[(i * 4)..], floatData[i]); } });
        }
        else if (dtype == OnnxType.Double && doubleData.Count > 0)
        {
            src = Bytes(doubleData.Count * 8, s => { for (int i = 0; i < doubleData.Count; i++) { BinaryPrimitives.WriteDoubleLittleEndian(s[(i * 8)..], doubleData[i]); } });
        }
        else if (dtype == OnnxType.Int64 && int64Data.Count > 0)
        {
            src = Bytes(int64Data.Count * 8, s => { for (int i = 0; i < int64Data.Count; i++) { BinaryPrimitives.WriteInt64LittleEndian(s[(i * 8)..], int64Data[i]); } });
        }
        else if (dtype is OnnxType.UInt64 or OnnxType.UInt32 && uint64Data.Count > 0)
        {
            int es = OnnxType.ElementSize(dtype);
            src = Bytes(uint64Data.Count * es, s => { for (int i = 0; i < uint64Data.Count; i++) { if (es == 8)
                    {
                        BinaryPrimitives.WriteInt64LittleEndian(s[(i * 8)..], uint64Data[i]);
                    }
                    else
                    {
                        BinaryPrimitives.WriteUInt32LittleEndian(s[(i * 4)..], (uint)uint64Data[i]);
                    }
                } });
        }
        else if (int32Data.Count > 0 && dtype is OnnxType.Int32 or OnnxType.Int16 or OnnxType.Int8 or OnnxType.UInt8 or OnnxType.UInt16 or OnnxType.Bool or OnnxType.Float16 or OnnxType.BFloat16)
        {
            int es = OnnxType.ElementSize(dtype);
            src = Bytes(int32Data.Count * es, s =>
            {
                for (int i = 0; i < int32Data.Count; i++)
                {
                    long v = int32Data[i];
                    switch (es)
                    {
                        case 1: s[i] = (byte)v; break;
                        case 2: BinaryPrimitives.WriteUInt16LittleEndian(s[(i * 2)..], (ushort)v); break;
                        default: BinaryPrimitives.WriteInt32LittleEndian(s[(i * 4)..], (int)v); break;
                    }
                }
            });
        }
        else if (elems == 0 && dtype != OnnxType.String)
        {
            src = new ArrayWeightSource([]);
        }
        else if (dtype != OnnxType.String)
        {
            throw new InvalidDataException($"Tensor '{name}' ({OnnxType.Name(dtype)}) carries no data.");
        }

        if (src is not null && dtype != OnnxType.String && src.ByteLength != OnnxType.ElementSize(dtype) * elems)
        {
            throw new InvalidDataException($"Tensor '{name}': {src.ByteLength} data bytes for {elems} {OnnxType.Name(dtype)} elements.");
        }

        return new OnnxTensor { Name = name, ElemType = dtype, Dims = dimsArr, Source = src };

        static ArrayWeightSource Bytes(int n, Action<Span<byte>> fill) { var b = new byte[n]; fill(b); return new ArrayWeightSource(b); }
    }
}
