namespace Sharplight.Shaders;

/// <summary>Code generation target for kernels.</summary>
public enum ShaderTarget
{
    /// <summary>SPIR-V for Vulkan compute (pointers = physical storage buffers / BDA).</summary>
    SpirV,
    /// <summary>CUDA C++ source (compile with NVRTC).</summary>
    CudaSource,
    /// <summary>PTX assembly.</summary>
    Ptx,
    /// <summary>Metal Shading Language source.</summary>
    Metal,
}
