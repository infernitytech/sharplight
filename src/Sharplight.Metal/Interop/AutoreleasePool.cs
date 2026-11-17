namespace Sharplight.Metal.Interop;

/// <summary>
/// An Objective-C autorelease pool for a scope: Metal returns command buffers, encoders, strings and errors autoreleased, and threads
/// of the .NET thread pool have no pool that would ever drain them. <c>using var pool = AutoreleasePool.Push();</c>
/// </summary>
internal readonly ref struct AutoreleasePool
{
    private readonly nint _handle;

    private AutoreleasePool(nint handle)
    {
        _handle = handle;
    }

    /// <summary>Opens a pool on the current thread.</summary>
    public static AutoreleasePool Push() => new(ObjCRuntime.AutoreleasePoolPush());

    /// <summary>Releases every object autoreleased since <see cref="Push"/>.</summary>
    public void Dispose()
    {
        if (_handle != 0)
        {
            ObjCRuntime.AutoreleasePoolPop(_handle);
        }
    }
}
