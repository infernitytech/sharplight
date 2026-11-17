using System.Runtime.InteropServices;

namespace Sharplight.Metal.Interop;

/// <summary>The C functions of <c>Metal.framework</c> the provider calls. Only callable on macOS.</summary>
internal static partial class MetalFramework
{
    private const string Library = "/System/Library/Frameworks/Metal.framework/Metal";

    /// <summary>Path of Foundation, loaded before the first class lookup (<c>NSString</c>, <c>MTLCompileOptions</c>).</summary>
    internal const string FoundationLibrary = "/System/Library/Frameworks/Foundation.framework/Foundation";

    /// <summary><c>MTLCopyAllDevices()</c>: an <c>NSArray</c> of every GPU (retained; the caller releases it).</summary>
    [LibraryImport(Library, EntryPoint = "MTLCopyAllDevices")]
    internal static partial nint CopyAllDevices();

    /// <summary><c>MTLCreateSystemDefaultDevice()</c> (retained; the caller releases it).</summary>
    [LibraryImport(Library, EntryPoint = "MTLCreateSystemDefaultDevice")]
    internal static partial nint CreateSystemDefaultDevice();
}
