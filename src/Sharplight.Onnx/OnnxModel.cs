using Sharplight.Models;

namespace Sharplight.Onnx;

/// <summary>ONNX TensorProto.DataType ids and helpers.</summary>
public static class OnnxType
{
    public const int Float = 1, UInt8 = 2, Int8 = 3, UInt16 = 4, Int16 = 5, Int32 = 6, Int64 = 7, String = 8, Bool = 9,
        Float16 = 10, Double = 11, UInt32 = 12, UInt64 = 13, Complex64 = 14, Complex128 = 15, BFloat16 = 16;

    public static int ElementSize(int t) => t switch
    {
        Float or Int32 or UInt32 => 4,
        UInt8 or Int8 or Bool => 1,
        UInt16 or Int16 or Float16 or BFloat16 => 2,
        Int64 or UInt64 or Double or Complex64 => 8,
        Complex128 => 16,
        _ => throw new NotSupportedException($"ONNX element type {t} has no fixed element size."),
    };

    public static bool IsFloat(int t) => t is Float or Float16 or Double or BFloat16;

    public static string Name(int t) => t switch
    {
        Float => "float", UInt8 => "uint8", Int8 => "int8", UInt16 => "uint16", Int16 => "int16", Int32 => "int32", Int64 => "int64",
        String => "string", Bool => "bool", Float16 => "float16", Double => "double", UInt32 => "uint32", UInt64 => "uint64",
        BFloat16 => "bfloat16", _ => $"type#{t}",
    };
}

/// <summary>A dimension of an ONNX value's shape: fixed value or symbolic parameter (both absent = unknown).</summary>
public readonly record struct OnnxDim(long? Value, string? Param)
{
    public bool IsKnown => Value is > 0 && Param is null;
    public override string ToString() => Param ?? Value?.ToString() ?? "?";
}

public sealed class OnnxValueInfo
{
    public required string Name { get; init; }
    /// <summary>Element type (<see cref="OnnxType"/>), 0 if the value is not a dense tensor.</summary>
    public int ElemType { get; init; }
    /// <summary>Null if the shape is not stated.</summary>
    public IReadOnlyList<OnnxDim>? Shape { get; init; }
    public override string ToString() => $"{Name}: {OnnxType.Name(ElemType)}[{(Shape is null ? "?" : string.Join(", ", Shape))}]";
}

/// <summary>Tensor payload: dims, element type and a byte source holding the little-endian dense data.</summary>
public sealed class OnnxTensor
{
    public string Name { get; init; } = "";
    public int ElemType { get; init; }
    public long[] Dims { get; init; } = [];
    /// <summary>Dense little-endian element data (row-major); null for string tensors.</summary>
    public IWeightSource? Source { get; init; }
    public long Elements { get { long n = 1; foreach (var d in Dims) { n *= d; } return n; } }
    public override string ToString() => $"{Name} {OnnxType.Name(ElemType)}[{string.Join(", ", Dims)}]";
}

public enum OnnxAttrType { Undefined = 0, Float = 1, Int = 2, String = 3, Tensor = 4, Graph = 5, Floats = 6, Ints = 7, Strings = 8, Tensors = 9, Graphs = 10 }

public sealed class OnnxAttribute
{
    public required string Name { get; init; }
    public OnnxAttrType Type { get; init; }
    public float F { get; init; }
    public long I { get; init; }
    public string? S { get; init; }
    public OnnxTensor? T { get; init; }
    public OnnxGraph? G { get; init; }
    public IReadOnlyList<float> Floats { get; init; } = [];
    public IReadOnlyList<long> Ints { get; init; } = [];
    public IReadOnlyList<string> Strings { get; init; } = [];
    public override string ToString() => $"{Name}:{Type}";
}

public sealed class OnnxNode
{
    public string Name { get; init; } = "";
    public required string OpType { get; init; }
    public string Domain { get; init; } = "";
    /// <summary>Input names; empty string marks an omitted optional input.</summary>
    public required IReadOnlyList<string> Inputs { get; init; }
    public required IReadOnlyList<string> Outputs { get; init; }
    public IReadOnlyDictionary<string, OnnxAttribute> Attributes { get; init; } = new Dictionary<string, OnnxAttribute>();

    public bool Has(string a) => Attributes.ContainsKey(a);
    public long Int(string a, long def) => Attributes.TryGetValue(a, out var x) ? x.I : def;
    public float Float(string a, float def) => Attributes.TryGetValue(a, out var x) ? x.F : def;
    public string Str(string a, string def) => Attributes.TryGetValue(a, out var x) && x.S is not null ? x.S : def;
    public long[]? Ints(string a) => Attributes.TryGetValue(a, out var x) ? [.. x.Ints] : null;
    public float[]? Floats(string a) => Attributes.TryGetValue(a, out var x) ? [.. x.Floats] : null;
    public OnnxTensor? Tensor(string a) => Attributes.TryGetValue(a, out var x) ? x.T : null;

    public override string ToString() => $"{OpType} {Name} ({string.Join(", ", Inputs)}) -> ({string.Join(", ", Outputs)})";
}

public sealed class OnnxGraph
{
    public string Name { get; init; } = "";
    public IReadOnlyList<OnnxNode> Nodes { get; init; } = [];
    public IReadOnlyList<OnnxTensor> Initializers { get; init; } = [];
    public IReadOnlyList<OnnxValueInfo> Inputs { get; init; } = [];
    public IReadOnlyList<OnnxValueInfo> Outputs { get; init; } = [];
    public IReadOnlyList<OnnxValueInfo> ValueInfo { get; init; } = [];
    public string DocString { get; init; } = "";
}

public sealed class OnnxModel
{
    public long IrVersion { get; init; }
    public string ProducerName { get; init; } = "";
    public string ProducerVersion { get; init; } = "";
    public string Domain { get; init; } = "";
    public long ModelVersion { get; init; }
    public string DocString { get; init; } = "";
    /// <summary>Opset versions per domain ("" = the default ai.onnx domain).</summary>
    public IReadOnlyDictionary<string, long> Opsets { get; init; } = new Dictionary<string, long>();
    public IReadOnlyDictionary<string, string> MetadataProps { get; init; } = new Dictionary<string, string>();
    public required OnnxGraph Graph { get; init; }
    /// <summary>Options consulted by <see cref="OnnxGraphAdapter"/> (input shape overrides, ...).</summary>
    public OnnxImportOptions Options { get; set; } = new();

    public long Opset(string domain = "") => Opsets.TryGetValue(domain, out var v) ? v
        : domain == "" && Opsets.TryGetValue("ai.onnx", out v) ? v : 0;
}

/// <summary>Options for importing/translating an ONNX model.</summary>
public sealed class OnnxImportOptions
{
    /// <summary>
    /// Full-shape overrides per graph input name, e.g. <c>["pixel_values"] = [-1, 3, 384, 384]</c>.
    /// Use a value ≤ 0 for the (symbolic) batch axis. Needed when the model has dynamic spatial dims.
    /// </summary>
    public Dictionary<string, long[]> InputShapes { get; init; } = new(StringComparer.Ordinal);
    /// <summary>
    /// Graph inputs pinned to compile-time constants (flat values, shape from <see cref="InputShapes"/> or the declared dims), e.g.
    /// <c>["sr"] = [16000]</c> or an <c>orig_im_size</c> that determines output shapes. They are not runtime inputs of the compiled graph.
    /// </summary>
    public Dictionary<string, double[]> ConstantInputs { get; init; } = new(StringComparer.Ordinal);
    /// <summary>Make the outermost dim of every input symbolic ("N") even if it is fixed in the file.</summary>
    public bool ForceSymbolicBatch { get; init; }
    /// <summary>Placement group size, in compute-heavy ops (Conv/MatMul/Gemm/BatchMatMul) per group.</summary>
    public int GroupSize { get; init; } = 8;
}
