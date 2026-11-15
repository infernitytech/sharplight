using System.Runtime.InteropServices;

namespace Sharplight.Studio.Platform;

/// <summary>What the running X11 window manager supports, read from the root window's <c>_NET_SUPPORTED</c> list.</summary>
internal static partial class X11WindowManager
{
    private const string LibX11 = "libX11.so.6";
    private const nint AnyPropertyType = 0;
    private const int Success = 0;

    /// <summary>
    /// The window manager moves and resizes windows on request (<c>_NET_WM_MOVERESIZE</c>), which the own title bar and resize edges
    /// rely on. False without an X display, without a window manager, or when the library cannot be loaded.
    /// </summary>
    public static bool SupportsMoveResize()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DISPLAY")))
        {
            return false;
        }

        try
        {
            return Supports("_NET_WM_MOVERESIZE");
        }
        catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
            return false;
        }
    }

    private static unsafe bool Supports(string feature)
    {
        nint display = XOpenDisplay(0);
        if (display == 0)
        {
            return false;
        }

        try
        {
            nint supported = XInternAtom(display, "_NET_SUPPORTED", onlyIfExists: 1);
            nint wanted = XInternAtom(display, feature, onlyIfExists: 1);
            if (supported == 0 || wanted == 0)
            {
                return false;
            }

            int status = XGetWindowProperty(display, XDefaultRootWindow(display), supported, 0, 4096, 0, AnyPropertyType,
                out _, out int format, out nuint count, out _, out nint data);
            if (status != Success || data == 0)
            {
                return false;
            }

            try
            {
                // Format-32 properties come back as an array of C longs.
                if (format != 32)
                {
                    return false;
                }

                var atoms = new ReadOnlySpan<nint>((void*)data, checked((int)count));
                return atoms.Contains(wanted);
            }
            finally
            {
                _ = XFree(data);
            }
        }
        finally
        {
            _ = XCloseDisplay(display);
        }
    }

    [LibraryImport(LibX11)]
    private static partial nint XOpenDisplay(nint name);

    [LibraryImport(LibX11)]
    private static partial int XCloseDisplay(nint display);

    [LibraryImport(LibX11, StringMarshalling = StringMarshalling.Utf8)]
    private static partial nint XInternAtom(nint display, string name, int onlyIfExists);

    [LibraryImport(LibX11)]
    private static partial nint XDefaultRootWindow(nint display);

    [LibraryImport(LibX11)]
    private static partial int XGetWindowProperty(nint display, nint window, nint property, nint offset, nint length, int delete, nint requestedType,
        out nint actualType, out int actualFormat, out nuint itemCount, out nuint bytesAfter, out nint data);

    [LibraryImport(LibX11)]
    private static partial int XFree(nint data);
}
