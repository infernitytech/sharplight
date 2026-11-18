namespace Sharplight.Cuda;

/// <summary>Options of the CUDA execution provider (<c>services.AddCudaProvider(options => ...)</c>).</summary>
public sealed class CudaProviderOptions
{
    /// <summary>
    /// Lists integrated GPUs (Jetson / GB10-class devices with memory shared with the host) as well as discrete ones. Default:
    /// <see langword="true"/>.
    /// </summary>
    public bool IncludeIntegrated { get; set; } = true;
}
