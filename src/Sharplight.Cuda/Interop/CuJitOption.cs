namespace Sharplight.Cuda.Interop;

/// <summary>The <c>CUjit_option</c> values the provider passes to the driver's linker.</summary>
internal enum CuJitOption
{
    InfoLogBuffer = 3,
    InfoLogBufferSizeBytes = 4,
    ErrorLogBuffer = 5,
    ErrorLogBufferSizeBytes = 6,
}
