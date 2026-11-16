using System.Reflection;
using System.Runtime.InteropServices;

namespace Sharplight.Video.Native;

/// <summary>
/// Locates and loads the native library: the application's normal probing (NuGet <c>runtimes/&lt;rid&gt;/native</c> assets), then the
/// <c>runtimes/&lt;rid&gt;/native</c> folder next to the assembly (development builds), then the <c>SHARPLIGHT_VIDEO_LIBRARY</c> path.
/// </summary>
public static class NativeVideoLibrary
{
    /// <summary>Environment variable holding an explicit path to the native library; it takes precedence.</summary>
    public const string PathVariable = "SHARPLIGHT_VIDEO_LIBRARY";

    private static readonly Lazy<string?> s_loadError = new(Probe);
    private static int s_resolverInstalled;

    /// <summary>Whether the native library for this platform was found and matches the expected ABI version.</summary>
    public static bool IsAvailable => s_loadError.Value is null;

    /// <summary>Why <see cref="IsAvailable"/> is false; <see langword="null"/> when the library loaded.</summary>
    public static string? UnavailableReason => s_loadError.Value;

    /// <summary>The version of the bundled codec library, for example <c>libvpx v1.15.2</c>; empty when unavailable.</summary>
    public static string VersionText => IsAvailable ? Marshal.PtrToStringUTF8(NativeMethods.slv_version_string()) ?? string.Empty : string.Empty;

    /// <summary>Throws <see cref="PlatformNotSupportedException"/> when the library is unavailable.</summary>
    internal static void ThrowIfUnavailable()
    {
        string? error = s_loadError.Value;
        if (error is not null)
        {
            throw new PlatformNotSupportedException(error);
        }
    }

    /// <summary>The text of the last native error of the calling thread.</summary>
    internal static string LastError => Marshal.PtrToStringUTF8(NativeMethods.slv_last_error()) ?? "unknown error";

    private static string? Probe()
    {
        if (Interlocked.Exchange(ref s_resolverInstalled, 1) == 0)
        {
            NativeLibrary.SetDllImportResolver(typeof(NativeVideoLibrary).Assembly, Resolve);
        }

        try
        {
            int abi = NativeMethods.slv_abi_version();
            return abi == NativeMethods.AbiVersion ? null : $"The native video library has ABI version {abi}, expected {NativeMethods.AbiVersion}.";
        }
        catch (Exception exception) when (exception is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
            return $"The native video library for {RuntimeIdentifier} could not be loaded: {exception.Message}";
        }
    }

    private static string RuntimeIdentifier
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

    private static string FileName => OperatingSystem.IsWindows() ? "sharplight_video.dll" : OperatingSystem.IsMacOS() ? "libsharplight_video.dylib" : "libsharplight_video.so";

    private static nint Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != NativeMethods.Library)
        {
            return 0;
        }

        string? explicitPath = Environment.GetEnvironmentVariable(PathVariable);
        if (!string.IsNullOrEmpty(explicitPath) && NativeLibrary.TryLoad(explicitPath, out nint explicitHandle))
        {
            return explicitHandle;
        }

        if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out nint handle))
        {
            return handle;
        }

        string candidate = Path.Combine(AppContext.BaseDirectory, "runtimes", RuntimeIdentifier, "native", FileName);
        return File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out handle) ? handle : 0;
    }
}
