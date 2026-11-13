namespace Sharplight.Shaders.Lowering;

/// <summary>Device address of buffer + byte offset.</summary>
/// <param name="Buffer">The buffer.</param>
/// <param name="ByteOffset">Offset from the buffer start.</param>
public sealed record PointerArg(BufferDecl Buffer, long ByteOffset) : KernelArg;
