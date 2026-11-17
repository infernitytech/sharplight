using System.Runtime.InteropServices;

namespace Sharplight.Metal.Interop;

/// <summary>
/// The Objective-C runtime functions the provider calls (<c>libobjc</c>). Every message send goes through <c>objc_msgSend</c>, declared
/// once per argument and return signature: on arm64 the callee reads its arguments with the C calling convention of the method's own
/// signature, so each shape needs its own typed import. Only callable on macOS.
/// </summary>
internal static unsafe partial class ObjCRuntime
{
    private const string Library = "/usr/lib/libobjc.A.dylib";

    [LibraryImport(Library, EntryPoint = "objc_getClass", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint GetClass(string name);

    [LibraryImport(Library, EntryPoint = "sel_registerName", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint RegisterSelector(string name);

    [LibraryImport(Library, EntryPoint = "objc_autoreleasePoolPush")]
    internal static partial nint AutoreleasePoolPush();

    [LibraryImport(Library, EntryPoint = "objc_autoreleasePoolPop")]
    internal static partial void AutoreleasePoolPop(nint pool);

    /// <summary><c>id f(id, SEL)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial nint Send(nint receiver, nint selector);

    /// <summary><c>id f(id, SEL, id)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial nint Send(nint receiver, nint selector, nint argument);

    /// <summary><c>id f(id, SEL, NSUInteger)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial nint SendIndex(nint receiver, nint selector, nuint argument);

    /// <summary><c>id f(id, SEL, NSUInteger, NSUInteger)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial nint Send(nint receiver, nint selector, nuint first, nuint second);

    /// <summary><c>id f(id, SEL, id, NSError**)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial nint Send(nint receiver, nint selector, nint argument, nint* error);

    /// <summary><c>id f(id, SEL, id, id, NSError**)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial nint Send(nint receiver, nint selector, nint first, nint second, nint* error);

    /// <summary><c>id f(id, SEL, void*, NSUInteger, NSUInteger, id)</c> (<c>newBufferWithBytesNoCopy:length:options:deallocator:</c>).</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial nint Send(nint receiver, nint selector, void* bytes, nuint length, nuint options, nint deallocator);

    /// <summary><c>void f(id, SEL)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial void SendVoid(nint receiver, nint selector);

    /// <summary><c>void f(id, SEL, id)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial void SendVoid(nint receiver, nint selector, nint argument);

    /// <summary><c>void f(id, SEL, NSUInteger)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial void SendVoidIndex(nint receiver, nint selector, nuint argument);

    /// <summary><c>void f(id, SEL, BOOL)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial void SendVoidBool(nint receiver, nint selector, byte argument);

    /// <summary><c>void f(id, SEL, const void*, NSUInteger, NSUInteger)</c> (<c>setBytes:length:atIndex:</c>).</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial void SendVoid(nint receiver, nint selector, void* bytes, nuint length, nuint index);

    /// <summary><c>void f(id, SEL, const id*, NSUInteger, NSUInteger)</c> (<c>useResources:count:usage:</c>).</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial void SendVoid(nint receiver, nint selector, nint* objects, nuint count, nuint usage);

    /// <summary><c>void f(id, SEL, MTLSize, MTLSize)</c> (<c>dispatchThreadgroups:threadsPerThreadgroup:</c>).</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial void SendVoid(nint receiver, nint selector, MTLSize first, MTLSize second);

    /// <summary><c>BOOL f(id, SEL, id, NSError**)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial byte SendBool(nint receiver, nint selector, nint argument, nint* error);

    /// <summary><c>id f(id, SEL, id, NSUInteger, id*, NSError**)</c> (<c>newComputePipelineStateWithDescriptor:options:reflection:error:</c>).</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial nint Send(nint receiver, nint selector, nint argument, nuint options, nint* reflection, nint* error);

    /// <summary><c>BOOL f(id, SEL)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial byte SendBool(nint receiver, nint selector);

    /// <summary><c>BOOL f(id, SEL, NSInteger)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial byte SendBool(nint receiver, nint selector, nint argument);

    /// <summary><c>NSUInteger / uint64_t f(id, SEL)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial ulong SendUInt64(nint receiver, nint selector);

    /// <summary><c>double f(id, SEL)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial double SendDouble(nint receiver, nint selector);

    /// <summary><c>MTLSize f(id, SEL)</c>.</summary>
    [LibraryImport(Library, EntryPoint = "objc_msgSend")]
    internal static partial MTLSize SendSize(nint receiver, nint selector);
}
