using System.Runtime.InteropServices;
using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>Resolves the FFmpeg.AutoGen bindings against the library handles <see cref="FfmpegLoader"/> opened (never a system FFmpeg).</summary>
internal sealed class FfmpegFunctionResolver : IFunctionResolver
{
    private readonly IReadOnlyDictionary<string, nint> _libraries;

    /// <param name="libraries">The loaded libraries by FFmpeg name (<c>avcodec</c>, ...).</param>
    public FfmpegFunctionResolver(IReadOnlyDictionary<string, nint> libraries)
    {
        _libraries = libraries;
    }

    /// <inheritdoc />
    public T GetFunctionDelegate<T>(string libraryName, string functionName, bool throwOnError = true)
    {
        if (_libraries.TryGetValue(libraryName, out nint library) && NativeLibrary.TryGetExport(library, functionName, out nint address))
        {
            return Marshal.GetDelegateForFunctionPointer<T>(address);
        }

        if (throwOnError)
        {
            throw new EntryPointNotFoundException($"'{functionName}' is not exported by the bundled {libraryName} library.");
        }

        return default!;
    }
}
