namespace Sharplight.OnnxRun;

/// <summary>The execution providers the runner can use.</summary>
internal enum OnnxProvider
{
    /// <summary>Managed CPU kernels.</summary>
    Cpu,

    /// <summary>Vulkan compute on a GPU.</summary>
    Vulkan,
}
