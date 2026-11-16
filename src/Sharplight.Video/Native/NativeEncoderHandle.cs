using Microsoft.Win32.SafeHandles;

namespace Sharplight.Video.Native;

/// <summary>Owns a native encoder (<c>slv_encoder*</c>).</summary>
internal sealed class NativeEncoderHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    /// <summary>Creates an empty handle; the marshaller fills it in.</summary>
    public NativeEncoderHandle()
        : base(ownsHandle: true)
    {
    }

    /// <inheritdoc />
    protected override bool ReleaseHandle()
    {
        NativeMethods.slv_encoder_destroy(handle);
        return true;
    }
}
