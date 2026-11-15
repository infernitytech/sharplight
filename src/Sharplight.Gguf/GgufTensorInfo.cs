using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>Describes one tensor stored in a GGUF file. Valid while the owning <see cref="GgufFile"/> is open.</summary>
public sealed class GgufTensorInfo
{
    /// <summary>The file holding the data, or null for a tensor read from a header alone (<see cref="FromHeader"/>).</summary>
    private readonly GgufFile? _file;

    internal GgufTensorInfo(GgufFile? file, string name, int ggmlType, DataType dataType, long[] ne, long relativeOffset, long absoluteOffset, long byteSize)
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

    /// <summary>
    /// A tensor known from a tensor table read before the file is available (<see cref="GgufHeaderProbe.ReadTensorTable"/>): its
    /// weights have the right names, types and shapes, and reading their data throws <see cref="GgufHeaderOnlyDataException"/>.
    /// </summary>
    /// <param name="entry">The tensor table entry.</param>
    /// <param name="dataType">Its element type (<see cref="GgmlTypes.TryGetDataType"/>).</param>
    internal static GgufTensorInfo FromHeader(GgufTensorEntry entry, DataType dataType)
    {
        var ne = entry.Dimensions.ToArray();
        return new GgufTensorInfo(null, entry.Name, entry.GgmlType, dataType, ne, 0, 0, GgufFile.ComputeByteSize(dataType, ne, entry.Name));
    }

    /// <summary>
    /// A zero-copy source over the tensor bytes inside the file mapping (a source without data for a header-only tensor); for the
    /// interleaved Q4_0 types a source that reads them as plain Q4_0 rows.
    /// </summary>
    internal IWeightSource CreateSource() => _file is null ? new GgufHeaderOnlySource(Name, ByteSize)
        : GgmlTypes.TryGetQ4_0Interleave(GgmlType, out int rows, out int chunk)
            ? new GgufInterleavedQ4_0Source(_file, Offset, ByteSize, Ne[0] / DataType.BlockSize, rows, chunk)
            : new GgufWeightSource(_file, Offset, ByteSize);

    /// <summary>A zero-copy weight (file name, file shape) over the tensor bytes.</summary>
    internal WeightInfo ToWeightInfo() => new() { Name = Name, Type = DataType, Shape = Shape, Source = CreateSource() };

    /// <inheritdoc/>
    public override string ToString() => $"{Name} {DataType}{Shape} @{Offset} ({ByteSize} B)";
}
