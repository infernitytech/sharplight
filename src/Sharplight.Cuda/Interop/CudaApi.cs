using System.Runtime.InteropServices;

namespace Sharplight.Cuda.Interop;

/// <summary>
/// The CUDA driver API (<c>libcuda.so.1</c> / <c>nvcuda.dll</c>, installed with the NVIDIA driver), as far as the provider uses it.
/// <see cref="CudaLibraries.EnsureResolver"/> must run before the first call. Every function returns a <c>CUresult</c>
/// (<see cref="Check"/>). Device pointers are <c>CUdeviceptr</c> (64-bit); contexts, streams, events, modules and functions are handles.
/// </summary>
internal static unsafe partial class CudaApi
{
    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuInit(uint flags);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuDriverGetVersion(int* version);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuGetErrorString(int result, byte** text);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuDeviceGetCount(int* count);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuDeviceGet(int* device, int ordinal);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuDeviceGetName(byte* name, int length, int device);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuDeviceTotalMem_v2")]
    public static partial int cuDeviceTotalMem(nuint* bytes, int device);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuDeviceGetAttribute(int* value, CuDeviceAttribute attribute, int device);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuDeviceGetUuid_v2")]
    public static partial int cuDeviceGetUuid(byte* uuid, int device);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuDevicePrimaryCtxRetain(nint* context, int device);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuDevicePrimaryCtxRelease_v2")]
    public static partial int cuDevicePrimaryCtxRelease(int device);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuCtxSetCurrent(nint context);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuMemGetInfo_v2")]
    public static partial int cuMemGetInfo(nuint* free, nuint* total);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuMemAlloc_v2")]
    public static partial int cuMemAlloc(ulong* pointer, nuint bytes);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuMemFree_v2")]
    public static partial int cuMemFree(ulong pointer);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuMemAllocHost_v2")]
    public static partial int cuMemAllocHost(void** pointer, nuint bytes);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuMemFreeHost(void* pointer);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuMemcpyHtoD_v2")]
    public static partial int cuMemcpyHtoD(ulong destination, void* source, nuint bytes);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuMemcpyDtoH_v2")]
    public static partial int cuMemcpyDtoH(void* destination, ulong source, nuint bytes);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuMemcpyHtoDAsync_v2")]
    public static partial int cuMemcpyHtoDAsync(ulong destination, void* source, nuint bytes, nint stream);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuMemcpyDtoHAsync_v2")]
    public static partial int cuMemcpyDtoHAsync(void* destination, ulong source, nuint bytes, nint stream);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuMemcpyDtoDAsync_v2")]
    public static partial int cuMemcpyDtoDAsync(ulong destination, ulong source, nuint bytes, nint stream);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuMemsetD8Async(ulong destination, byte value, nuint count, nint stream);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuStreamCreate(nint* stream, uint flags);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuStreamDestroy_v2")]
    public static partial int cuStreamDestroy(nint stream);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuStreamSynchronize(nint stream);

    /// <summary><c>CU_STREAM_CAPTURE_MODE_THREAD_LOCAL</c>: only this thread's unsafe calls are prohibited during a capture.</summary>
    public const int CaptureModeThreadLocal = 1;

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuStreamBeginCapture_v2")]
    public static partial int cuStreamBeginCapture(nint stream, int mode);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuStreamEndCapture(nint stream, nint* graph);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuGraphInstantiateWithFlags(nint* exec, nint graph, ulong flags);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuGraphLaunch(nint exec, nint stream);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuGraphExecDestroy(nint exec);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuGraphDestroy(nint graph);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuEventCreate(nint* evt, uint flags);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuEventDestroy_v2")]
    public static partial int cuEventDestroy(nint evt);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuEventRecord(nint evt, nint stream);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuEventSynchronize(nint evt);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuStreamWaitEvent(nint stream, nint evt, uint flags);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuMemHostRegister_v2")]
    public static partial int cuMemHostRegister(void* pointer, nuint bytes, uint flags);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuMemHostUnregister(void* pointer);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuEventElapsedTime(float* milliseconds, nint start, nint end);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuModuleLoadData(nint* module, void* image);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuModuleUnload(nint module);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuModuleGetFunction(nint* function, nint module, byte* name);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuFuncGetAttribute(int* value, int attribute, nint function);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuFuncSetAttribute(nint function, int attribute, int value);

    /// <summary>
    /// Encodes a tiled tensor map (<c>CUtensorMap</c>, 128 bytes) for tensor copies into shared memory
    /// (<c>cp.async.bulk.tensor</c>, compute capability 9.0 and later).
    /// </summary>
    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuTensorMapEncodeTiled(void* tensorMap, int dataType, uint rank, ulong globalAddress, ulong* globalDims,
        ulong* globalStrides, uint* boxDims, uint* elementStrides, int interleave, int swizzle, int l2Promotion, int oobFill);

    /// <summary><c>CU_TENSOR_MAP_DATA_TYPE_UINT8</c>.</summary>
    public const int TensorMapUInt8 = 0;

    /// <summary><c>CU_TENSOR_MAP_SWIZZLE_128B</c>.</summary>
    public const int TensorMapSwizzle128 = 3;

    /// <summary><c>CU_TENSOR_MAP_L2_PROMOTION_L2_256B</c>.</summary>
    public const int TensorMapL2Promotion256 = 3;

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuLaunchKernel(nint function, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ,
        uint sharedMemoryBytes, nint stream, void** parameters, void** extra);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuLinkCreate_v2")]
    public static partial int cuLinkCreate(uint optionCount, CuJitOption* options, void** values, nint* state);

    [LibraryImport(CudaLibraries.Driver, EntryPoint = "cuLinkAddData_v2")]
    public static partial int cuLinkAddData(nint state, int type, void* data, nuint size, byte* name, uint optionCount, CuJitOption* options, void** values);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuLinkComplete(nint state, void** cubin, nuint* size);

    [LibraryImport(CudaLibraries.Driver)]
    public static partial int cuLinkDestroy(nint state);

    /// <summary><c>CU_JIT_INPUT_PTX</c>: the input type of PTX for <see cref="cuLinkAddData"/>.</summary>
    public const int JitInputPtx = 1;

    /// <summary><c>CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK</c>.</summary>
    public const int FunctionMaxThreadsPerBlock = 0;

    /// <summary><c>CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES</c>.</summary>
    public const int FunctionMaxDynamicSharedSizeBytes = 8;

    /// <summary><c>CU_FUNC_ATTRIBUTE_NUM_REGS</c>.</summary>
    public const int FunctionRegisters = 4;

    /// <summary><c>CU_EVENT_DEFAULT</c>: an event that records timing.</summary>
    public const uint EventDefault = 0;

    /// <summary><c>CU_EVENT_DISABLE_TIMING</c>.</summary>
    public const uint EventDisableTiming = 2;

    /// <summary><c>CU_MEMHOSTREGISTER_READ_ONLY</c>: the device only reads the registered memory (allows read-only file mappings).</summary>
    public const uint MemHostRegisterReadOnly = 8;

    /// <summary>Throws a <see cref="CudaException"/> for a failed call.</summary>
    /// <param name="result">The <c>CUresult</c> of the call.</param>
    /// <param name="what">The call, for the message.</param>
    public static void Check(int result, string what)
    {
        if (result != (int)CuResult.Success)
        {
            throw new CudaException(result, $"CUDA {what} failed: {ErrorText(result)}");
        }
    }

    /// <summary>The driver's name and description of a <c>CUresult</c>.</summary>
    public static string ErrorText(int result)
    {
        byte* text = null;
        string description = cuGetErrorString(result, &text) == 0 && text != null ? Marshal.PtrToStringUTF8((nint)text) ?? "" : "unknown error";
        return $"{description} ({result})";
    }
}
