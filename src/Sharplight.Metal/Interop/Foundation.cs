using System.Runtime.InteropServices;

namespace Sharplight.Metal.Interop;

/// <summary>Helpers for the few Foundation objects the provider handles (strings, arrays, errors). Callers hold an autorelease pool.</summary>
internal static unsafe class Foundation
{
    private static readonly nint s_stringClass = ObjCRuntime.GetClass("NSString");

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
