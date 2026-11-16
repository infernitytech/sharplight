namespace Sharplight.Audio.Native;

/// <summary>Owns a native encoder (<c>sla_encoder*</c>).</summary>
internal sealed class NativeEncoderHandle : NativeHandle
{
    /// <summary>Wraps a handle returned by the native library.</summary>
    /// <param name="handle">The native pointer.</param>
    public NativeEncoderHandle(nint handle)
        : base(handle)
    {
    }

    /// <inheritdoc />
    protected override bool ReleaseHandle()
    {
        NativeMethods.sla_encoder_close(handle);
        return true;
    }
}
