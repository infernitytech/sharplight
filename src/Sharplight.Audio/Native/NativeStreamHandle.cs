namespace Sharplight.Audio.Native;

/// <summary>Owns a native stream (<c>sla_stream*</c>).</summary>
internal sealed class NativeStreamHandle : NativeHandle
{
    /// <summary>Wraps a handle returned by the native library.</summary>
    /// <param name="handle">The native pointer.</param>
    public NativeStreamHandle(nint handle)
        : base(handle)
    {
    }

    /// <inheritdoc />
    protected override bool ReleaseHandle()
    {
        NativeMethods.sla_stream_close(handle);
        return true;
    }
}
