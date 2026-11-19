namespace Sharplight.Cuda;

/// <summary>A library GEMM a launch runs on cuBLASLt instead of a kernel (see <see cref="Shaders.LibraryKernel"/>).</summary>
/// <param name="Type">cuBLASLt data type of x and w.</param>
/// <param name="N">Output columns (weight rows).</param>
/// <param name="K">Inner dimension.</param>
internal sealed record CudaLibraryGemm(int Type, long N, long K);
