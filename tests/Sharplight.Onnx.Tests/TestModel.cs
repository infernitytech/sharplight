namespace Sharplight.Onnx.Tests;

/// <summary>Assembles ModelProto bytes for graph-level tests.</summary>
internal sealed class TestModel
{
    private readonly List<byte[]> _nodes = [];
    private readonly List<byte[]> _inits = [];
    private readonly List<byte[]> _inputs = [];
    private readonly List<byte[]> _outputs = [];
    private readonly List<(string, long)> _opsets = [("", 17)];
    private readonly List<(string, string)> _meta = [];
    public string GraphName = "test";

    public TestModel Opset(long v, string domain = "")
    {
        _opsets.Clear();
        _opsets.Add((domain, v));
        return this;
    }
    public TestModel Meta(string k, string v)
    {
        _meta.Add((k, v));
        return this;
    }

    public static ProtoWriter Attr(string name, long i) => new ProtoWriter().Str(1, name).Int(3, i).Int(20, 2);
    public static ProtoWriter AttrF(string name, float f) => new ProtoWriter().Str(1, name).Float(2, f).Int(20, 1);
    public static ProtoWriter AttrS(string name, string s) => new ProtoWriter().Str(1, name).Str(4, s).Int(20, 3);
    public static ProtoWriter AttrInts(string name, params long[] v)
    {
        var w = new ProtoWriter().Str(1, name);
        foreach (var x in v)
        {
            w.Int(8, x);
        }
        return w.Int(20, 7);
    }
    public static ProtoWriter AttrTensor(string name, ProtoWriter t) => new ProtoWriter().Str(1, name).Msg(5, t).Int(20, 4);

    public TestModel Node(string op, string[] ins, string[] outs, params ProtoWriter[] attrs)
    {
        var w = new ProtoWriter();
        foreach (var i in ins)
        {
            w.Str(1, i);
        }

        foreach (var o in outs)
        {
            w.Str(2, o);
        }

        w.Str(3, outs[0] + "_node").Str(4, op);
        foreach (var a in attrs)
        {
            w.Msg(5, a);
        }

        _nodes.Add(w.ToArray());
        return this;
    }

    public static ProtoWriter TensorProto(string name, int type, long[] dims, byte[]? raw = null)
    {
        var w = new ProtoWriter();
        foreach (var d in dims)
        {
            w.Int(1, d);
        }

        w.Int(2, type);
        if (name != "")
        {
            w.Str(8, name);
        }

        if (raw is not null)
        {
            w.Bytes(9, raw);
        }

        return w;
    }

    public TestModel InitF32(string name, long[] dims, float[] data)
    {
        var raw = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
        _inits.Add(TensorProto(name, 1, dims, raw).ToArray());
        return this;
    }

    /// <summary>float tensor stored with the typed float_data field (no raw_data).</summary>
    public TestModel InitF32Typed(string name, long[] dims, float[] data)
    {
        _inits.Add(TensorProto(name, 1, dims).PackedFloats(4, data).ToArray());
        return this;
    }

    public TestModel InitI64(string name, long[] dims, params long[] data)
    {
        var raw = new byte[data.Length * 8];
        Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
        _inits.Add(TensorProto(name, 7, dims, raw).ToArray());
        return this;
    }

    public TestModel InitRaw(ProtoWriter tensor)
    {
        _inits.Add(tensor.ToArray());
        return this;
    }

    public static ProtoWriter ValueInfo(string name, int type, params object[] dims)
    {
        var shape = new ProtoWriter();
        foreach (var d in dims)
        {
            shape.Msg(1, d is string s ? new ProtoWriter().Str(2, s) : new ProtoWriter().Int(1, Convert.ToInt64(d)));
        }

        var tt = new ProtoWriter().Int(1, type).Msg(2, shape);
        return new ProtoWriter().Str(1, name).Msg(2, new ProtoWriter().Msg(1, tt));
    }

    public TestModel Input(string name, params object[] dims)
    {
        _inputs.Add(ValueInfo(name, 1, dims).ToArray());
        return this;
    }
    public TestModel Output(string name, params object[] dims)
    {
        _outputs.Add(ValueInfo(name, 1, dims).ToArray());
        return this;
    }

    public byte[] ToBytes()
    {
        var g = new ProtoWriter();
        foreach (var n in _nodes)
        {
            g.Bytes(1, n);
        }

        g.Str(2, GraphName);
        foreach (var i in _inits)
        {
            g.Bytes(5, i);
        }

        foreach (var i in _inputs)
        {
            g.Bytes(11, i);
        }

        foreach (var o in _outputs)
        {
            g.Bytes(12, o);
        }

        var m = new ProtoWriter().Int(1, 9).Str(2, "unit-test").Str(3, "1.2").Str(6, "doc text").Msg(7, g);
        foreach (var (d, v) in _opsets)
        {
            m.Msg(8, new ProtoWriter().Str(1, d).Int(2, v));
        }

        foreach (var (k, v) in _meta)
        {
            m.Msg(14, new ProtoWriter().Str(1, k).Str(2, v));
        }

        return m.ToArray();
    }

    public Models.ModelSource Source(OnnxImportOptions? options = null)
    {
        var model = OnnxParser.Parse(ToBytes());
        if (options is not null)
        {
            model.Options = options;
        }

        return OnnxImporter.FromModel(model);
    }

    public Models.ModelDefinition Build(OnnxImportOptions? options = null) => OnnxGraphAdapter.Build(Source(options));
}
