using Sharplight.Models;
using Xunit;

namespace Sharplight.Onnx.Tests;

public class ProtoReaderTests
{
    private static ulong Var(byte[] b) { var r = new ProtoReader(b); return r.ReadVarint(); }

    [Theory]
    [InlineData(0UL)]
    [InlineData(1UL)]
    [InlineData(127UL)]
    [InlineData(128UL)]
    [InlineData(300UL)]
    [InlineData(16384UL)]
    [InlineData(uint.MaxValue)]
    [InlineData(ulong.MaxValue)]
    public void Varint_RoundTrips(ulong v) => Assert.Equal(v, Var(new ProtoWriter().Varint(v).ToArray()));

    [Fact]
    public void NegativeInt64_UsesTenByteVarint()
    {
        var b = new ProtoWriter().Int(1, -5).ToArray();
        var r = new ProtoReader(b);
        Assert.True(r.Next(out int f, out var w));
        Assert.Equal(1, f); Assert.Equal(WireType.Varint, w);
        Assert.Equal(-5L, r.ReadInt64());
        Assert.False(r.Next(out _, out _));
    }

    [Fact]
    public void Fields_AllWireTypes_AndSkip()
    {
        var b = new ProtoWriter().Int(1, 150).Float(2, 1.5f).Str(3, "héllo").Bytes(4, [1, 2, 3]).Int(5, 7).ToArray();
        var r = new ProtoReader(b);
        r.Next(out _, out var w); Assert.Equal(150, r.ReadInt32());
        r.Next(out _, out w); Assert.Equal(WireType.Fixed32, w); Assert.Equal(1.5f, r.ReadFloat());
        r.Next(out _, out w); Assert.Equal("héllo", r.ReadString());
        r.Next(out int f, out w); Assert.Equal(4, f); r.Skip(w);
        r.Next(out f, out w); Assert.Equal(5, f); Assert.Equal(7, r.ReadInt32());
        Assert.True(r.AtEnd);
    }

    [Fact]
    public void ReadBytes_ReportsAbsoluteOffset()
    {
        var payload = new byte[] { 9, 8, 7 };
        var b = new ProtoWriter().Int(1, 1).Bytes(2, payload).ToArray();
        var r = new ProtoReader(b);
        r.Next(out _, out var w); r.ReadVarint();
        r.Next(out _, out w);
        var s = r.ReadBytes(out int off);
        Assert.Equal(payload, s.ToArray());
        Assert.Equal(payload, b.AsSpan(off, 3).ToArray());
    }

    [Fact]
    public void NestedMessage_IsBounded()
    {
        var inner = new ProtoWriter().Int(1, 42).Int(2, 43);
        var b = new ProtoWriter().Msg(1, inner).Int(2, 99).ToArray();
        var r = new ProtoReader(b);
        r.Next(out _, out _);
        var m = r.ReadMessage();
        m.Next(out _, out _); Assert.Equal(42, m.ReadInt32());
        m.Next(out _, out _); Assert.Equal(43, m.ReadInt32());
        Assert.False(m.Next(out _, out _));
        r.Next(out int f, out _); Assert.Equal(2, f); Assert.Equal(99, r.ReadInt32());
    }

    [Fact]
    public void PackedAndUnpackedRepeated()
    {
        var b = new ProtoWriter().PackedInts(1, [1, 2, -3, 1000]).Int(1, 77).PackedFloats(2, [0.5f, 2f]).Float(2, 3f).ToArray();
        var r = new ProtoReader(b);
        var ints = new List<long>(); var floats = new List<float>();
        while (r.Next(out int f, out var w))
        {
            if (f == 1)
            {
                r.ReadRepeatedInt64(w, ints);
            }
            else
            {
                r.ReadRepeatedFloat(w, floats);
            }
        }

        Assert.Equal([1L, 2, -3, 1000, 77], ints);
        Assert.Equal([0.5f, 2f, 3f], floats);
    }

    [Fact]
    public void Truncated_Throws()
    {
        Assert.Throws<InvalidDataException>(() => { var r = new ProtoReader(new byte[] { 0x80 }); r.ReadVarint(); });
        Assert.Throws<InvalidDataException>(() => { var r = new ProtoReader(new byte[] { 0x0A, 0x05, 1, 2 }); r.Next(out _, out _); r.ReadBytes(out _); });
        Assert.Throws<InvalidDataException>(() => { var r = new ProtoReader(new byte[] { 0x0D, 1, 2 }); r.Next(out _, out _); r.ReadFixed32(); });
        Assert.Throws<InvalidDataException>(() => { var r = new ProtoReader(new byte[] { 0x00 }); r.Next(out _, out _); });   // field number 0
    }
}

public class OnnxParserTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "onnx-tests-" + Guid.NewGuid().ToString("N"));
    public OnnxParserTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    [Fact]
    public void ParsesModelStructure()
    {
        var bytes = new TestModel()
            .Meta("author", "me")
            .InitF32("w", [2, 3], [1, 2, 3, 4, 5, 6])
            .Input("x", "N", 3).Output("y", "N", 2)
            .Node("Gemm", ["x", "w"], ["y"], TestModel.AttrInts("kernel_shape", 3, 3), TestModel.Attr("transB", 1), TestModel.AttrF("alpha", 0.5f), TestModel.AttrS("mode", "foo"))
            .ToBytes();
        var m = OnnxParser.Parse(bytes);
        Assert.Equal(9, m.IrVersion);
        Assert.Equal("unit-test", m.ProducerName); Assert.Equal("1.2", m.ProducerVersion); Assert.Equal("doc text", m.DocString);
        Assert.Equal(17, m.Opset());
        Assert.Equal("me", m.MetadataProps["author"]);
        var g = m.Graph;
        Assert.Equal("test", g.Name);
        var n = Assert.Single(g.Nodes);
        Assert.Equal("Gemm", n.OpType); Assert.Equal(["x", "w"], n.Inputs); Assert.Equal(["y"], n.Outputs);
        Assert.Equal(1, n.Int("transB", 0)); Assert.Equal(0.5f, n.Float("alpha", 1)); Assert.Equal("foo", n.Str("mode", ""));
        Assert.Equal([3L, 3], n.Ints("kernel_shape"));
        Assert.Equal(7, n.Int("missing", 7));
        var t = Assert.Single(g.Initializers);
        Assert.Equal(OnnxType.Float, t.ElemType); Assert.Equal([2L, 3], t.Dims);
        var buf = new byte[24]; t.Source!.Read(0, buf);
        Assert.Equal([1f, 2, 3, 4, 5, 6], System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(buf).ToArray());
        var inp = Assert.Single(g.Inputs);
        Assert.Equal("x", inp.Name); Assert.Equal("N", inp.Shape![0].Param); Assert.Equal(3, inp.Shape[1].Value);
        Assert.False(inp.Shape[0].IsKnown); Assert.True(inp.Shape[1].IsKnown);
    }

    [Fact]
    public void TypedTensorData_IsConvertedToBytes()
    {
        var bytes = new TestModel()
            .InitF32Typed("f", [3], [1.5f, -2f, 4f])
            .InitRaw(TestModel.TensorProto("i", OnnxType.Int64, [2]).PackedInts(7, [5, -6]))
            .InitRaw(TestModel.TensorProto("i32", OnnxType.Int32, [2]).PackedInts(5, [-7, 9]))
            .InitRaw(TestModel.TensorProto("b", OnnxType.Bool, [3]).PackedInts(5, [1, 0, 1]))
            .Input("x", 1).Output("y", 1).Node("Identity", ["x"], ["y"]).ToBytes();
        var m = OnnxParser.Parse(bytes);
        var by = m.Graph.Initializers.ToDictionary(t => t.Name);
        Assert.Equal([1.5, -2, 4], OnnxConstOf(by["f"]).Floats());
        Assert.Equal([5L, -6], OnnxConstOf(by["i"]).Ints());
        Assert.Equal([-7L, 9], OnnxConstOf(by["i32"]).Ints());
        Assert.Equal([1L, 0, 1], OnnxConstOf(by["b"]).Ints());
    }

    private static OnnxConst OnnxConstOf(OnnxTensor t) => OnnxConst.FromTensor(t);

    [Fact]
    public void ByteCountMismatch_Throws()
    {
        var bytes = new TestModel().InitRaw(TestModel.TensorProto("w", 1, [4], new byte[8])).Input("x", 1).Output("y", 1).Node("Identity", ["x"], ["y"]).ToBytes();
        var e = Assert.Throws<InvalidDataException>(() => OnnxParser.Parse(bytes));
        Assert.Contains("'w'", e.Message);
    }

    [Fact]
    public void ExternalData_MappedZeroCopy_AndTraversalRejected()
    {
        var data = new float[16]; for (int i = 0; i < 16; i++)
        {
            data[i] = i;
        }

        var dataBytes = new byte[64 + 32]; Buffer.BlockCopy(data, 0, dataBytes, 32, 64);
        File.WriteAllBytes(Path.Combine(_dir, "m.onnx.data"), dataBytes);

        ProtoWriter Ext(string loc, long off, long len) => TestModel.TensorProto("w", 1, [4, 4])
            .Int(14, 1)
            .Msg(13, new ProtoWriter().Str(1, "location").Str(2, loc))
            .Msg(13, new ProtoWriter().Str(1, "offset").Str(2, off.ToString()))
            .Msg(13, new ProtoWriter().Str(1, "length").Str(2, len.ToString()));

        var good = new TestModel().InitRaw(Ext("m.onnx.data", 32, 64)).Input("x", 4, 4).Output("y", 4, 4).Node("Add", ["x", "w"], ["y"]).ToBytes();
        File.WriteAllBytes(Path.Combine(_dir, "m.onnx"), good);
        using var src = DisposeWith(OnnxImporter.Open(Path.Combine(_dir, "m.onnx")));
        var w = src.Weights["w"];
        Assert.NotEqual(0, w.Source.TryGetPointer());
        Assert.Equal(64, w.Source.ByteLength);
        Assert.Equal(data, w.ReadAll().Select((_, i) => i).Where(i => i % 4 == 0).Select(i => BitConverter.ToSingle(w.ReadAll(), i)).ToArray());

        var bad = new TestModel().InitRaw(Ext("../evil.bin", 0, 64)).Input("x", 4, 4).Output("y", 4, 4).Node("Add", ["x", "w"], ["y"]).ToBytes();
        File.WriteAllBytes(Path.Combine(_dir, "bad.onnx"), bad);
        Assert.Throws<InvalidDataException>(() => OnnxImporter.Open(Path.Combine(_dir, "bad.onnx")));

        var oob = new TestModel().InitRaw(Ext("m.onnx.data", 60, 64)).Input("x", 4, 4).Output("y", 4, 4).Node("Add", ["x", "w"], ["y"]).ToBytes();
        File.WriteAllBytes(Path.Combine(_dir, "oob.onnx"), oob);
        Assert.Throws<InvalidDataException>(() => OnnxImporter.Open(Path.Combine(_dir, "oob.onnx")));
    }

    private sealed class Owned(ModelSource s) : IDisposable
    {
        public ModelSource S = s;
        public IReadOnlyDictionary<string, WeightInfo> Weights => S.Weights;
        public void Dispose()
        {
            foreach (var r in S.OwnedResources)
            {
                r.Dispose();
            }
        }
    }
    private static Owned DisposeWith(ModelSource s) => new(s);

    [Fact]
    public void Importer_MetadataAndZeroCopyRawWeights()
    {
        var path = Path.Combine(_dir, "a.onnx");
        File.WriteAllBytes(path, new TestModel().Meta("k", "v").InitF32("w", [2, 2], [1, 2, 3, 4]).Input("x", "N", 2).Output("y", "N", 2).Node("Add", ["x", "w"], ["y"]).ToBytes());
        var imp = new OnnxImporter();
        Assert.True(imp.CanOpen(path));
        var src = ((Sharplight.Models.IModelImporter)imp).Open(path);
        try
        {
            Assert.Equal("onnx", src.Architecture); Assert.Equal("onnx", src.Format);
            Assert.IsType<OnnxModel>(src.FormatData);
            Assert.Equal("unit-test", src.Get<string>("onnx.producer_name"));
            Assert.Equal(17L, src.Get<long>("onnx.opset"));
            Assert.Equal("v", src.Get<string>("onnx.meta.k"));
            Assert.Equal("doc text", src.Get<string>("onnx.doc_string"));
            var w = src.Weights["w"];
            Assert.Equal("f32", w.Type.Name); Assert.Equal("[2, 2]", w.Shape.ToString());
            Assert.NotEqual(0, w.Source.TryGetPointer());   // memory-mapped, no copy
        }
        finally
        {
            foreach (var r in src.OwnedResources)
            {
                r.Dispose();
            }
        }
    }

    [Fact]
    public void CanOpen_RejectsNonOnnx()
    {
        var imp = new OnnxImporter();
        var txt = Path.Combine(_dir, "x.onnx"); File.WriteAllText(txt, "this is not protobuf at all, really");
        Assert.False(imp.CanOpen(txt));
        var wrongExt = Path.Combine(_dir, "model.bin"); File.WriteAllBytes(wrongExt, new TestModel().Input("x", 1).Output("y", 1).Node("Identity", ["x"], ["y"]).ToBytes());
        Assert.False(imp.CanOpen(wrongExt));
        Assert.False(imp.CanOpen(Path.Combine(_dir, "missing.onnx")));
        var trunc = Path.Combine(_dir, "t.onnx"); var full = new TestModel().Input("x", 1).Output("y", 1).Node("Identity", ["x"], ["y"]).ToBytes();
        File.WriteAllBytes(trunc, full[..(full.Length - 5)]);
        Assert.False(imp.CanOpen(trunc));
    }

    [Fact]
    public void Register_MakesImporterAndAdapterDiscoverable()
    {
        OnnxImporter.Register();
        var path = Path.Combine(_dir, "r.onnx");
        File.WriteAllBytes(path, new TestModel().Input("x", "N", 4).Output("y", "N", 4).Node("Relu", ["x"], ["y"]).ToBytes());
        var src = ModelImporterRegistry.Open(path);
        try
        {
            var def = ModelAdapterRegistry.Build(new ModelSourceSet { Primary = src });
            Assert.Equal("onnx", def.Architecture);
            Assert.Single(def.Graphs);
            Assert.Equal("main", def.Graphs.Keys.Single());
        }
        finally
        {
            foreach (var r in src.OwnedResources)
            {
                r.Dispose();
            }
        }
    }
}
