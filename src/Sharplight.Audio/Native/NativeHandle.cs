using Microsoft.Win32.SafeHandles;

namespace Sharplight.Audio.Native;

/// <summary>Base of the owning wrappers of native handles: keeps the handle alive while a call is in flight.</summary>
internal abstract class NativeHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    protected NativeHandle(nint handle)
        : base(ownsHandle: true)
    {
        SetHandle(handle);
    }

    /// <summary>Pins the handle for the duration of a native call; dispose the lease afterwards.</summary>
    /// <exception cref="ObjectDisposedException">The handle was closed.</exception>
    public HandleLease Lease() => new(this);
}
