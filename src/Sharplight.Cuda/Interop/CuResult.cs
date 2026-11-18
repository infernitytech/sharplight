namespace Sharplight.Cuda.Interop;

/// <summary>The <c>CUresult</c> codes the provider handles specially (every other value is reported through <c>cuGetErrorString</c>).</summary>
internal enum CuResult
{
    Success = 0,
    NotInitialized = 3,
    NoDevice = 100,
    InvalidDevice = 101,
    NoBinaryForGpu = 209,
    InvalidPtx = 218,
    JitCompilerNotFound = 221,
    UnsupportedPtxVersion = 222,
    InsufficientDriver = 35,
}
