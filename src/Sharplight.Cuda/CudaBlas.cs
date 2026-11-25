using System.Collections.Concurrent;
using Sharplight.Cuda.Interop;
using Sharplight.Runtime;

namespace Sharplight.Cuda;

/// <summary>
/// The library GEMMs of a CUDA device (<see cref="Runtime.DeviceFeatures.LibraryGemm"/>) on cuBLASLt from the optional
/// <c>Sharplight.Cuda.Blas</c> package: y [rows, N] f32 = x [rows, K] · wᵀ with x and w (N rows of K values) in bf16 or f16, f32
/// accumulation. cuBLASLt is column-major, so the product is computed as yᵀ = w · xᵀ: w is the transposed K x N operand A, x the K x rows
/// operand B. One handle and workspace per device; a plan (descriptors and the heuristic's algorithm) per shape and row count.
/// </summary>
internal sealed unsafe class CudaBlas : IDisposable
{
    /// <summary>Workspace the algorithms may use.</summary>
    private const long WorkspaceBytes = 32L << 20;

    private static readonly Lazy<ulong> s_version = new(LoadVersion);

    private readonly CudaDevice _device;
    private readonly nint _handle;
    private readonly CudaBuffer _workspace;
    private readonly ConcurrentDictionary<(int Type, long N, long K, long Rows, int Alignment), Plan> _plans = new();
    private bool _disposed;

    private CudaBlas(CudaDevice device, nint handle, CudaBuffer workspace)
    {
        _device = device;
        _handle = handle;
        _workspace = workspace;
    }

    /// <summary>The cuBLASLt version (e.g. 130100 for 13.1.0), or 0 when the library is not installed.</summary>
    public static ulong Version => s_version.Value;

    /// <summary>True when cuBLASLt can be loaded.</summary>
    public static bool Available => Version != 0;

    /// <summary>Creates the cuBLASLt handle and workspace of <paramref name="device"/>, or returns null when cuBLASLt is not installed.</summary>
    public static CudaBlas? TryCreate(CudaDevice device)
    {
        if (!Available)
        {
            return null;
        }

        device.Enter();
        nint handle;
        if (CublasLtApi.cublasLtCreate(&handle) != 0)
        {
            return null;
        }

        try
        {
            return new CudaBlas(device, handle, device.Allocate(WorkspaceBytes, "cuBLASLt workspace"));
        }
        catch (DeviceOutOfMemoryException)
        {
            // No room for the workspace on a small or crowded device: the products stay on the generated kernels.
            CublasLtApi.cublasLtDestroy(handle);
            return null;
        }
    }

    /// <summary>The cuBLASLt data type of a library kernel's operand type, or null when the library GEMM does not take it.</summary>
    public static int? DataType(string type) => type switch
    {
        "bf16" => CublasLtApi.R16BF,
        "f16" => CublasLtApi.R16F,
        _ => null,
    };

    /// <summary>Queues y = x · wᵀ on <paramref name="stream"/>.</summary>
    /// <exception cref="CudaException">cuBLASLt has no algorithm for the shape or the call failed.</exception>
    public void Gemm(nint stream, int type, long n, long k, long rows, ulong x, ulong w, ulong y)
    {
        if (rows == 0)
        {
            return;
        }

        // Algorithms assume 256-byte aligned matrices unless told otherwise: the plan is chosen for the operands' actual alignment.
        int alignment = (int)Math.Min(256UL, Lowest(x | w | y));
        var plan = _plans.GetOrAdd((type, n, k, rows, alignment), key => CreatePlan(key.Type, key.N, key.K, key.Rows, key.Alignment));
        float alpha = 1;
        float beta = 0;
        fixed (byte* algo = plan.Algorithm)
        {
            int status = CublasLtApi.cublasLtMatmul(_handle, plan.Desc, &alpha, w, plan.A, x, plan.B, &beta, y, plan.D, y, plan.D, algo,
                _workspace.Pointer, (nuint)WorkspaceBytes, stream);
            if (status != 0)
            {
                throw new CudaException(status, $"cuBLASLt GEMM [{k}→{n}] x {rows} rows failed (status {status}).");
            }
        }
    }

    private static ulong Lowest(ulong value) => value == 0 ? 256 : value & (~value + 1);

    private Plan CreatePlan(int type, long n, long k, long rows, int alignment)
    {
        _device.Enter();
        nint desc;
        nint a;
        nint b;
        nint d;
        nint preference;
        Check(CublasLtApi.cublasLtMatmulDescCreate(&desc, CublasLtApi.Compute32F, CublasLtApi.R32F), "matmul descriptor");
        int transposed = CublasLtApi.OpT;
        Check(CublasLtApi.cublasLtMatmulDescSetAttribute(desc, CublasLtApi.DescTransA, &transposed, sizeof(int)), "transpose A");
        Check(CublasLtApi.cublasLtMatrixLayoutCreate(&a, type, (ulong)k, (ulong)n, k), "layout of w");
        Check(CublasLtApi.cublasLtMatrixLayoutCreate(&b, type, (ulong)k, (ulong)rows, k), "layout of x");
        Check(CublasLtApi.cublasLtMatrixLayoutCreate(&d, CublasLtApi.R32F, (ulong)n, (ulong)rows, n), "layout of y");
        Check(CublasLtApi.cublasLtMatmulPreferenceCreate(&preference), "preference");
        ulong workspace = WorkspaceBytes;
        Check(CublasLtApi.cublasLtMatmulPreferenceSetAttribute(preference, CublasLtApi.PreferenceMaxWorkspaceBytes, &workspace, sizeof(ulong)), "workspace preference");
        uint minimum = (uint)alignment;
        for (int matrix = 0; matrix < 4; matrix++)
        {
            Check(CublasLtApi.cublasLtMatmulPreferenceSetAttribute(preference, CublasLtApi.PreferenceMinAlignmentA + matrix, &minimum, sizeof(uint)), "alignment preference");
        }

        var result = new byte[CublasLtApi.HeuristicResultBytes];
        int returned;
        fixed (byte* results = result)
        {
            Check(CublasLtApi.cublasLtMatmulAlgoGetHeuristic(_handle, desc, a, b, d, d, preference, 1, results, &returned), "algorithm heuristic");
        }

        CublasLtApi.cublasLtMatmulPreferenceDestroy(preference);
        if (returned == 0)
        {
            throw new CudaException(15, $"cuBLASLt has no algorithm for the GEMM [{k}→{n}] x {rows} rows.");
        }

        return new Plan(desc, a, b, d, result[..CublasLtApi.AlgoBytes]);
    }

    private static void Check(int status, string what)
    {
        if (status != 0)
        {
            throw new CudaException(status, $"cuBLASLt {what} failed (status {status}).");
        }
    }

    private static ulong LoadVersion()
    {
        CudaLibraries.EnsureResolver();
        try
        {
            return CublasLtApi.cublasLtGetVersion();
        }
        catch (DllNotFoundException)
        {
            return 0;
        }
        catch (EntryPointNotFoundException)
        {
            return 0;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (!_device.IsDisposed)
        {
            _device.Enter();
        }

        foreach (var plan in _plans.Values)
        {
            CublasLtApi.cublasLtMatrixLayoutDestroy(plan.A);
            CublasLtApi.cublasLtMatrixLayoutDestroy(plan.B);
            CublasLtApi.cublasLtMatrixLayoutDestroy(plan.D);
            CublasLtApi.cublasLtMatmulDescDestroy(plan.Desc);
        }

        CublasLtApi.cublasLtDestroy(_handle);
        _device.Free(_workspace);
    }

    /// <summary>The descriptors and algorithm of one GEMM shape.</summary>
    private sealed record Plan(nint Desc, nint A, nint B, nint D, byte[] Algorithm);
}
