namespace Sharplight.Audio.Native;

/// <summary>Result codes of the native library (<c>sla_result</c>).</summary>
internal enum NativeResult
{
    Ok = 0,
    InvalidArgument = -1,
    OutOfMemory = -2,
    NoBackend = -3,
    Io = -4,
    Unsupported = -5,
    Decode = -6,
    Device = -7,
    InvalidState = -8,
    BufferTooSmall = -9,
}
