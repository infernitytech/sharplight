using System.Runtime.InteropServices;

namespace Sharplight.Metal.Interop;

/// <summary>Helpers for the few Foundation objects the provider handles (strings, arrays, errors). Callers hold an autorelease pool.</summary>
internal static unsafe class Foundation
{
    private static readonly nint s_stringClass = ObjCRuntime.GetClass("NSString");
    private static readonly nint s_urlClass = ObjCRuntime.GetClass("NSURL");
    private static readonly nint s_arrayClass = ObjCRuntime.GetClass("NSArray");

    /// <summary>An autoreleased file <c>NSURL</c> for <paramref name="path"/>.</summary>
    public static nint FileUrl(string path) => ObjCRuntime.Send(s_urlClass, Selectors.FileUrlWithPath, String(path));

    /// <summary>An autoreleased <c>NSArray</c> holding <paramref name="obj"/>.</summary>
    public static nint ArrayOf(nint obj) => ObjCRuntime.Send(s_arrayClass, Selectors.ArrayWithObject, obj);

    /// <summary>A new (retained) instance of the Objective-C class <paramref name="className"/> (<c>[[C alloc] init]</c>).</summary>
    public static nint New(string className) => ObjCRuntime.Send(ObjCRuntime.Send(ObjCRuntime.GetClass(className), Selectors.Alloc), Selectors.Init);

    /// <summary>An autoreleased <c>NSString</c> with the text of <paramref name="value"/>.</summary>
    public static nint String(string value)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(value + "\0");
        fixed (byte* p = bytes)
        {
            return ObjCRuntime.Send(s_stringClass, Selectors.StringWithUtf8, (nint)p);
        }
    }

    /// <summary>The text of an <c>NSString</c>, or an empty string for nil.</summary>
    public static string ToManaged(nint nsString)
        => nsString == 0 ? "" : Marshal.PtrToStringUTF8(ObjCRuntime.Send(nsString, Selectors.Utf8String)) ?? "";

    /// <summary>The localized description of an <c>NSError</c>, or <paramref name="fallback"/> for nil.</summary>
    public static string ErrorText(nint error, string fallback)
        => error == 0 ? fallback : ToManaged(ObjCRuntime.Send(error, Selectors.LocalizedDescription));

    /// <summary>Sends <c>release</c> to a retained object (nil is ignored).</summary>
    public static void Release(nint obj)
    {
        if (obj != 0)
        {
            ObjCRuntime.SendVoid(obj, Selectors.Release);
        }
    }
}
