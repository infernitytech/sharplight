using System.Runtime.InteropServices;

namespace Sharplight.Audio.Native;

/// <summary>Reference on a <see cref="NativeHandle"/> held while a native call uses it.</summary>
internal readonly ref struct HandleLease
{
    private readonly SafeHandle _owner;

    /// <summary>Takes a reference on <paramref name="owner"/>.</summary>
    /// <param name="owner">The handle to keep alive.</param>
    public HandleLease(SafeHandle owner)
    {
        bool added = false;
        owner.DangerousAddRef(ref added);
        _owner = owner;
        Value = owner.DangerousGetHandle();
    }

    /// <summary>The raw native handle.</summary>
    public nint Value { get; }

    /// <summary>Releases the reference.</summary>
    public void Dispose() => _owner.DangerousRelease();
}
