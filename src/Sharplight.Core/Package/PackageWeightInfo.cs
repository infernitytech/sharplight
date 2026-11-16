using Sharplight.Tensors;

namespace Sharplight.Package;

/// <summary>One entry of a package's weight index.</summary>
/// <param name="Name">Canonical weight name.</param>
/// <param name="Type">Stored element type (canonical source type, e.g. a quantized block type).</param>
/// <param name="Shape">Fixed shape.</param>
/// <param name="Offset">Absolute file offset of the bytes (64 KiB aligned).</param>
/// <param name="ByteSize">Stored size in bytes.</param>
public sealed record PackageWeightInfo(string Name, DataType Type, Shape Shape, long Offset, long ByteSize);
