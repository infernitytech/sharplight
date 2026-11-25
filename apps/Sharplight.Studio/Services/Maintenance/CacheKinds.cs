namespace Sharplight.Studio.Services.Maintenance;

/// <summary>The caches that can be cleared (all rebuild on demand).</summary>
[Flags]
internal enum CacheKinds
{
    /// <summary>Nothing.</summary>
    None = 0,

    /// <summary>Compiled kernels: the compilation cache with CUDA PTX and cubin and SPIR-V device stages.</summary>
    Kernels = 1,

    /// <summary>Extracted shader sources.</summary>
    Shaders = 2,

    /// <summary>Other cache files (media caches, thumbnails, decoded audio).</summary>
    Media = 4,

    /// <summary>Every cache.</summary>
    All = Kernels | Shaders | Media,
}
