using System.Reflection;
using System.Runtime.InteropServices;

namespace Sharplight.Cuda.Interop;

/// <summary>
/// Locates the native CUDA libraries: the driver API from the NVIDIA driver installation (<c>libcuda.so.1</c>, <c>nvcuda.dll</c>), and
/// NVRTC with its builtins library and the floating-point headers from the <c>Sharplight.Cuda.Runtime</c> package (next to the
/// application or in its <c>runtimes/&lt;rid&gt;/native</c> folder), or from the directory named by <c>SHARPLIGHT_CUDA_RUNTIME_DIR</c>.
/// Nothing is taken from a CUDA toolkit installation.
/// </summary>
internal static class CudaLibraries
{
    /// <summary>Library name of the driver API in the <c>LibraryImport</c> declarations.</summary>
    public const string Driver = "sharplight-cuda-driver";

    /// <summary>Library name of NVRTC in the <c>LibraryImport</c> declarations.</summary>
    public const string Nvrtc = "sharplight-cuda-nvrtc";

    /// <summary>Library name of cuBLASLt (optional package <c>Sharplight.Cuda.Blas</c>) in the <c>LibraryImport</c> declarations.</summary>
    public const string BlasLt = "sharplight-cuda-cublaslt";

    private static readonly Lock s_gate = new();
    private static bool s_registered;
    private static string? s_nvrtcDirectory;

    /// <summary>Registers the import resolver of this assembly (once per process).</summary>
    public static void EnsureResolver()
    {
        lock (s_gate)
        {
            if (s_registered)
            {
                return;
            }

            NativeLibrary.SetDllImportResolver(typeof(CudaLibraries).Assembly, Resolve);
            s_registered = true;
        }
    }

    /// <summary>The directory NVRTC was loaded from (the floating-point headers ship next to it), or null before it was loaded.</summary>
    public static string? NvrtcDirectory => s_nvrtcDirectory;

    /// <summary>File names of the driver library on this OS.</summary>
    private static string[] DriverNames => OperatingSystem.IsWindows() ? ["nvcuda.dll"] : ["libcuda.so.1", "libcuda.so"];

    /// <summary>NVRTC and the file name prefix of its builtins library on this OS, newest major version first.</summary>
    private static (string Nvrtc, string BuiltinsPrefix)[] NvrtcNames => OperatingSystem.IsWindows()
        ? [("nvrtc64_130_0.dll", "nvrtc-builtins64_130"), ("nvrtc64_120_0.dll", "nvrtc-builtins64_12")]
        : [("libnvrtc.so.13", "libnvrtc-builtins.so.13"), ("libnvrtc.so.12", "libnvrtc-builtins.so.12")];

    private static nint Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        return name switch
        {
            Driver => LoadFirst(DriverNames.Select(n => (n, (string?)null))),
            Nvrtc => LoadNvrtc(),
            BlasLt => LoadFirst(BlasDirectories().SelectMany(d => BlasLtNames.Select(n => (n, (string?)d)))),
            _ => 0,
        };
    }

    private static nint LoadFirst(IEnumerable<(string Name, string? Directory)> candidates)
    {
        foreach (var (name, directory) in candidates)
        {
            string path = directory is null ? name : Path.Combine(directory, name);
            if (NativeLibrary.TryLoad(path, out nint handle))
            {
                return handle;
            }
        }

        return 0;
    }

    /// <summary>File names of cuBLASLt on this OS.</summary>
    private static string[] BlasLtNames => OperatingSystem.IsWindows() ? ["cublasLt64_13.dll"] : ["libcublasLt.so.13"];

    /// <summary>
    /// Directories searched for cuBLASLt, in order: <c>SHARPLIGHT_CUDA_BLAS_DIR</c>, then those of the runtime package (the
    /// <c>Sharplight.Cuda.Blas</c> package puts its files next to the application like <c>Sharplight.Cuda.Runtime</c>).
    /// </summary>
    public static IEnumerable<string> BlasDirectories()
    {
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_CUDA_BLAS_DIR") is { Length: > 0 } configured)
        {
            yield return configured;
        }

        foreach (string directory in RuntimeDirectories())
        {
            yield return directory;
        }
    }

    /// <summary>Directories searched for NVRTC, in order.</summary>
    public static IEnumerable<string> RuntimeDirectories()
    {
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_CUDA_RUNTIME_DIR") is { Length: > 0 } configured)
        {
            yield return configured;
        }

        string baseDirectory = AppContext.BaseDirectory;
        yield return baseDirectory;
        yield return Path.Combine(baseDirectory, "runtimes", RuntimeInformation.RuntimeIdentifier, "native");
        string arch = RuntimeInformation.OSArchitecture == Architecture.Arm64 ? "arm64" : "x64";
        yield return Path.Combine(baseDirectory, "runtimes", (OperatingSystem.IsWindows() ? "win-" : "linux-") + arch, "native");
    }

    /// <summary>
    /// Loads NVRTC from the first runtime directory that has it. The builtins library is loaded first by its full path: NVRTC opens it by
    /// file name, which then resolves to the already loaded library without a search path.
    /// </summary>
    private static nint LoadNvrtc()
    {
        foreach (string directory in RuntimeDirectories())
        {
            foreach (var (nvrtc, builtinsPrefix) in NvrtcNames)
            {
                string path = Path.Combine(directory, nvrtc);
                if (!File.Exists(path))
                {
                    continue;
                }

                // The builtins file carries the minor version (libnvrtc-builtins.so.13.0); match it by prefix.
                foreach (string candidate in Directory.EnumerateFiles(directory, builtinsPrefix + "*"))
                {
                    NativeLibrary.TryLoad(candidate, out _);
                }

                if (NativeLibrary.TryLoad(path, out nint handle))
                {
                    s_nvrtcDirectory = directory;
                    return handle;
                }
            }
        }

        return 0;
    }
}
