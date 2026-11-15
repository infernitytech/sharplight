namespace Sharplight.Audio.Native;

/// <summary>Owns a native decoder (<c>sla_decoder*</c>).</summary>
internal sealed class NativeDecoderHandle : NativeHandle
{
    /// <summary>Wraps a handle returned by the native library.</summary>
    /// <param name="handle">The native pointer.</param>
    public NativeDecoderHandle(nint handle)
        : base(handle)
    {
    }

    /// <inheritdoc />
    protected override bool ReleaseHandle()
    {
        NativeMethods.sla_decoder_close(handle);
        return true;
    }
}
