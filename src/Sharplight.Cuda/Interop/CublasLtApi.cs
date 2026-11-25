using System.Runtime.InteropServices;

namespace Sharplight.Cuda.Interop;

/// <summary>
/// The cuBLASLt entry points the library GEMM uses (<c>cublasLt.h</c>, CUDA 13), resolved through <see cref="CudaLibraries"/> from the
/// optional <c>Sharplight.Cuda.Blas</c> package. Every call returns a <c>cublasStatus_t</c> (0 = success).
/// </summary>
internal static unsafe partial class CublasLtApi
{
    /// <summary><c>CUBLAS_COMPUTE_32F</c>.</summary>
    public const int Compute32F = 68;

    /// <summary><c>CUDA_R_32F</c>.</summary>
    public const int R32F = 0;

    /// <summary><c>CUDA_R_16F</c>.</summary>
    public const int R16F = 2;

    /// <summary><c>CUDA_R_16BF</c>.</summary>
    public const int R16BF = 14;

    /// <summary><c>CUBLASLT_MATMUL_DESC_TRANSA</c>.</summary>
    public const int DescTransA = 3;

    /// <summary><c>CUBLASLT_MATMUL_DESC_TRANSB</c>.</summary>
    public const int DescTransB = 4;

    /// <summary><c>CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES</c>.</summary>
    public const int PreferenceMaxWorkspaceBytes = 1;

    /// <summary><c>CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES</c>; B, C and D follow (6, 7, 8).</summary>
    public const int PreferenceMinAlignmentA = 5;

    /// <summary><c>CUBLAS_OP_T</c>.</summary>
    public const int OpT = 1;

    /// <summary>Size of <c>cublasLtMatmulHeuristicResult_t</c>: the 64-byte algorithm, the workspace size, state, waves and reserved words.</summary>
    public const int HeuristicResultBytes = 96;

    /// <summary>Size of <c>cublasLtMatmulAlgo_t</c> at the start of a heuristic result.</summary>
    public const int AlgoBytes = 64;

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial nuint cublasLtGetVersion();

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtCreate(nint* handle);

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtDestroy(nint handle);

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtMatmulDescCreate(nint* desc, int computeType, int scaleType);

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtMatmulDescDestroy(nint desc);

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtMatmulDescSetAttribute(nint desc, int attribute, void* value, nuint size);

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtMatrixLayoutCreate(nint* layout, int type, ulong rows, ulong cols, long ld);

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtMatrixLayoutDestroy(nint layout);

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtMatmulPreferenceCreate(nint* preference);

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtMatmulPreferenceDestroy(nint preference);

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtMatmulPreferenceSetAttribute(nint preference, int attribute, void* value, nuint size);

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtMatmulAlgoGetHeuristic(nint handle, nint desc, nint a, nint b, nint c, nint d, nint preference, int requested,
        byte* results, int* returned);

    [LibraryImport(CudaLibraries.BlasLt)]
    public static partial int cublasLtMatmul(nint handle, nint desc, void* alpha, ulong a, nint aLayout, ulong b, nint bLayout, void* beta, ulong c,
        nint cLayout, ulong d, nint dLayout, void* algo, ulong workspace, nuint workspaceBytes, nint stream);
}
