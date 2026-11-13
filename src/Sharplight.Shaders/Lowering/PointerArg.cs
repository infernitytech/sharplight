namespace Sharplight.Shaders.Lowering;

/// <summary>Device address of buffer + byte offset.</summary>
public sealed record PointerArg(BufferDecl Buffer, long ByteOffset) : KernelArg;
