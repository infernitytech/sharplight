using System.Runtime.InteropServices;

namespace Sharplight.Cuda.Interop;

/// <summary>
/// The NVIDIA runtime compiler (NVRTC: CUDA C++ to PTX or cubin), shipped by the <c>Sharplight.Cuda.Runtime</c> package next to the
/// application. <see cref="CudaLibraries.EnsureResolver"/> must run before the first call. Functions return an <c>nvrtcResult</c> (0 = success).
/// </summary>
internal static unsafe partial class NvrtcApi
{
    [LibraryImport(CudaLibraries.Nvrtc)]
    public static partial int nvrtcVersion(int* major, int* minor);

    [LibraryImport(CudaLibraries.Nvrtc)]
    public static partial byte* nvrtcGetErrorString(int result);

    [LibraryImport(CudaLibraries.Nvrtc)]
    public static partial int nvrtcCreateProgram(nint* program, byte* source, byte* name, int headerCount, byte** headers, byte** includeNames);

    [LibraryImport(CudaLibraries.Nvrtc)]
    public static partial int nvrtcCompileProgram(nint program, int optionCount, byte** options);

    [LibraryImport(CudaLibraries.Nvrtc)]
    public static partial int nvrtcDestroyProgram(nint* program);

    [LibraryImport(CudaLibraries.Nvrtc)]
    public static partial int nvrtcGetProgramLogSize(nint program, nuint* size);

    [LibraryImport(CudaLibraries.Nvrtc)]
    public static partial int nvrtcGetProgramLog(nint program, byte* log);

    [LibraryImport(CudaLibraries.Nvrtc)]
    public static partial int nvrtcGetPTXSize(nint program, nuint* size);

    [LibraryImport(CudaLibraries.Nvrtc)]
    public static partial int nvrtcGetPTX(nint program, byte* ptx);

    [LibraryImport(CudaLibraries.Nvrtc)]
    public static partial int nvrtcGetCUBINSize(nint program, nuint* size);

    [LibraryImport(CudaLibraries.Nvrtc)]
    public static partial int nvrtcGetCUBIN(nint program, byte* cubin);

    /// <summary>Throws a <see cref="CudaException"/> for a failed call.</summary>
    /// <param name="result">The <c>nvrtcResult</c> of the call.</param>
    /// <param name="what">The call, for the message.</param>
    public static void Check(int result, string what)
    {
        if (result != 0)
        {
            throw new CudaException(result, $"NVRTC {what} failed: {Marshal.PtrToStringUTF8((nint)nvrtcGetErrorString(result))} ({result})");
        }
    }
}
