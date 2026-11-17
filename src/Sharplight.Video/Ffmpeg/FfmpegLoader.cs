using System.Runtime.InteropServices;
using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// Loads Sharplight's own FFmpeg build (native/ffmpeg) once per process and binds FFmpeg.AutoGen to it. The libraries are opened by full
/// path in dependency order from, in turn: the <see cref="BundledFfmpeg.DirectoryVariable"/> directory, the application directory (where
/// publishing puts the <c>runtimes/&lt;rid&gt;/native</c> assets), <c>runtimes/&lt;rid&gt;/native</c> under it (development builds), then the host's native probing of the NuGet assets. A system FFmpeg is never used: the major versions must be
/// those the bindings were generated for.
/// </summary>
internal static class FfmpegLoader
{
    /// <summary>The libraries in load order (each only depends on the ones before it).</summary>
    public static readonly string[] Libraries = ["avutil", "swresample", "swscale", "avcodec", "avformat", "avfilter"];

    private static readonly Lazy<LoadResult> s_result = new(Load, LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>The outcome of the one load attempt of this process.</summary>
    public static LoadResult Result => s_result.Value;

    /// <summary>The platform's file name of one library, for example <c>libavcodec.so.63</c>.</summary>
    /// <param name="library">The FFmpeg library name (<c>avcodec</c>).</param>
    /// <returns>The file name.</returns>
    public static string FileName(string library)
    {
        int major = ffmpeg.LibraryVersionMap[library];
        if (OperatingSystem.IsWindows())
        {
            return $"{library}-{major}.dll";
        }

        return OperatingSystem.IsMacOS() ? $"lib{library}.{major}.dylib" : $"lib{library}.so.{major}";
    }

    /// <summary>The runtime identifier of the process, for example <c>linux-x64</c>.</summary>
    public static string RuntimeIdentifier
    {
        get
        {
            string os = OperatingSystem.IsWindows() ? "win" : OperatingSystem.IsMacOS() ? "osx" : "linux";
            string architecture = RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.Arm64 => "arm64",
                Architecture.X64 => "x64",
                var other => other.ToString().ToLowerInvariant(),
            };
            return $"{os}-{architecture}";
        }
    }

    /// <summary>The directories searched for the libraries, in order.</summary>
    public static IReadOnlyList<string> SearchDirectories()
    {
        var directories = new List<string>();
        string? configured = Environment.GetEnvironmentVariable(BundledFfmpeg.DirectoryVariable);
        if (!string.IsNullOrWhiteSpace(configured))
        {
            directories.Add(configured);
        }

        string native = Path.Combine("runtimes", RuntimeIdentifier, "native");
        directories.Add(AppContext.BaseDirectory);
        directories.Add(Path.Combine(AppContext.BaseDirectory, native));
        return [.. directories.Distinct(StringComparer.Ordinal)];
    }

    private static LoadResult Load()
    {
        string? directory = SearchDirectories().FirstOrDefault(d => File.Exists(Path.Combine(d, FileName("avutil"))));
        var handles = new Dictionary<string, nint>(StringComparer.Ordinal);
        foreach (string library in Libraries)
        {
            string fileName = FileName(library);
            nint handle;
            bool loaded = directory is not null
                ? NativeLibrary.TryLoad(Path.Combine(directory, fileName), out handle)
                : NativeLibrary.TryLoad(fileName, typeof(FfmpegLoader).Assembly, null, out handle);
            if (!loaded)
            {
                string where = directory ?? string.Join(", ", SearchDirectories());
                return new LoadResult(null, $"The bundled FFmpeg library {fileName} for {RuntimeIdentifier} could not be loaded (searched {where}).");
            }

            handles[library] = handle;
        }

        DynamicallyLoadedBindings.FunctionResolver = new FfmpegFunctionResolver(handles);
        DynamicallyLoadedBindings.Initialize();
        string? mismatch = CheckVersions();
        if (mismatch is not null)
        {
            return new LoadResult(null, mismatch);
        }

        ffmpeg.av_log_set_level(ffmpeg.AV_LOG_ERROR);
        return new LoadResult(directory, null);
    }

    private static string? CheckVersions()
    {
        (string Name, uint Version)[] versions =
        [
            ("avutil", ffmpeg.avutil_version()),
            ("swresample", ffmpeg.swresample_version()),
            ("swscale", ffmpeg.swscale_version()),
            ("avcodec", ffmpeg.avcodec_version()),
            ("avformat", ffmpeg.avformat_version()),
            ("avfilter", ffmpeg.avfilter_version()),
        ];
        foreach ((string name, uint version) in versions)
        {
            int major = (int)(version >> 16);
            if (major != ffmpeg.LibraryVersionMap[name])
            {
                return $"The FFmpeg library {name} has major version {major}, the bindings expect {ffmpeg.LibraryVersionMap[name]}.";
            }
        }

        return null;
    }

    /// <summary>The outcome of loading.</summary>
    /// <param name="Directory">The directory the libraries came from (<see langword="null"/> when found by the host's probing or not loaded).</param>
    /// <param name="Error">Why loading failed, or <see langword="null"/>.</param>
    internal sealed record LoadResult(string? Directory, string? Error);
}
