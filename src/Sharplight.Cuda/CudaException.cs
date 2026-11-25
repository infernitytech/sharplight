namespace Sharplight.Cuda;

/// <summary>A CUDA driver call failed.</summary>
/// <param name="result">The <c>CUresult</c> code.</param>
/// <param name="message">What failed, with the driver's description of the code.</param>
public sealed class CudaException(int result, string message) : Exception(message)
{
    /// <summary>The <c>CUresult</c> code of the failed call.</summary>
    public int Result { get; } = result;
}
