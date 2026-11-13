using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>Describes one tensor stored in a GGUF file.</summary>
public sealed class GgufTensorInfo
{
    private readonly GgufFile _file;

    internal GgufTensorInfo(GgufFile file, string name, int ggmlType, DataType dataType, long[] ne, long relativeOffset, long absoluteOffset, long byteSize)
    {
        _file = file; Name = name; GgmlType = ggmlType; DataType = dataType; Ne = ne;
        RelativeOffset = relativeOffset; Offset = absoluteOffset; ByteSize = byteSize;
        Shape = new Shape(ne.Reverse().Select(n => (Dim)n));
    }

    public string Name { get; }
    /// <summary>Raw ggml_type id.</summary>
    public int GgmlType { get; }
    public DataType DataType { get; }
    /// <summary>Shape, outermost dimension first (reverse of <see cref="Ne"/>).</summary>
    public Shape Shape { get; }
    /// <summary>ggml dimensions, innermost first.</summary>
    public long[] Ne { get; }
    /// <summary>Offset relative to the start of the data section.</summary>
    public long RelativeOffset { get; }
    /// <summary>Absolute offset in the file.</summary>
    public long Offset { get; }
    public long ByteSize { get; }
    public long ElementCount => Shape.Elements();
    public GgufFile File => _file;

    /// <summary>Pointer to the tensor bytes inside the file mapping (valid while the file is open).</summary>
    public unsafe nint Pointer => (nint)(_file.BasePointer + Offset);

    public IWeightSource CreateSource() => new GgufWeightSource(_file, Offset, ByteSize);

    public WeightInfo ToWeightInfo() => new() { Name = Name, Type = DataType, Shape = Shape, Source = CreateSource() };

    public override string ToString() => $"{Name} {DataType}{Shape} @{Offset} ({ByteSize} B)";
}

internal sealed unsafe class GgufWeightSource(GgufFile file, long offset, long length) : IWeightSource
{
    public long ByteLength => length;

    public void Read(long off, Span<byte> destination)
    {
        if (off < 0 || off + destination.Length > length)
        {
            throw new ArgumentOutOfRangeException(nameof(off));
        }

        file.ThrowIfDisposed();
        new ReadOnlySpan<byte>(file.BasePointer + offset + off, destination.Length).CopyTo(destination);
    }

    public nint TryGetPointer()
    {
        file.ThrowIfDisposed();
        return (nint)(file.BasePointer + offset);
    }
}
