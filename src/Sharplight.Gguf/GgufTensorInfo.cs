using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>Describes one tensor stored in a GGUF file. Valid while the owning <see cref="GgufFile"/> is open.</summary>
public sealed class GgufTensorInfo
{
    private readonly GgufFile _file;

    internal GgufTensorInfo(GgufFile file, string name, int ggmlType, DataType dataType, long[] ne, long relativeOffset, long absoluteOffset, long byteSize)
    {
        _file = file;
        Name = name;
        GgmlType = ggmlType;
        DataType = dataType;
        Ne = ne;
        RelativeOffset = relativeOffset;
        Offset = absoluteOffset;
        ByteSize = byteSize;
        Shape = new Shape(ne.Reverse().Select(n => (Dim)n));
    }

    /// <summary>The tensor's name as stored in the file (llama.cpp naming).</summary>
    public string Name { get; }

    /// <summary>Raw ggml_type id.</summary>
    internal int GgmlType { get; }

    /// <summary>The element type (possibly block-quantized).</summary>
    public DataType DataType { get; }

    /// <summary>Shape, outermost dimension first (reverse of <see cref="Ne"/>).</summary>
    public Shape Shape { get; }

    /// <summary>ggml dimensions, innermost first.</summary>
    internal long[] Ne { get; }

    /// <summary>Offset relative to the start of the data section.</summary>
    internal long RelativeOffset { get; }

    /// <summary>Absolute offset of the tensor data in the file, in bytes.</summary>
    public long Offset { get; }

    /// <summary>Size of the tensor data in bytes.</summary>
    public long ByteSize { get; }

    /// <summary>A zero-copy source over the tensor bytes inside the file mapping.</summary>
    internal IWeightSource CreateSource() => new GgufWeightSource(_file, Offset, ByteSize);

    /// <summary>A zero-copy weight (file name, file shape) over the tensor bytes.</summary>
    internal WeightInfo ToWeightInfo() => new() { Name = Name, Type = DataType, Shape = Shape, Source = CreateSource() };

    /// <inheritdoc/>
    public override string ToString() => $"{Name} {DataType}{Shape} @{Offset} ({ByteSize} B)";
}
