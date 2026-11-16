namespace Sharplight.Audio.Native;

/// <summary>Owns a native context (<c>sla_context*</c>).</summary>
internal sealed class NativeContextHandle : NativeHandle
{
    /// <summary>Wraps a handle returned by the native library.</summary>
    /// <param name="handle">The native pointer.</param>
    public NativeContextHandle(nint handle)
        : base(handle)
    {
    }

    /// <inheritdoc />
    protected override bool ReleaseHandle()
    {
        NativeMethods.sla_context_destroy(handle);
        return true;
    }
}
