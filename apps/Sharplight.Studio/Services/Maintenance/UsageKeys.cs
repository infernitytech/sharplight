namespace Sharplight.Studio.Services.Maintenance;

/// <summary>Keys of the usage slices (<see cref="StorageUsageItem.Key"/>).</summary>
internal static class UsageKeys
{
    /// <summary>Model files (weights and everything else in the packages, without their precompiled artifacts).</summary>
    public const string Models = "models";

    /// <summary>Precompiled artifact sets stored inside the model packages.</summary>
    public const string PackageArtifacts = "package-artifacts";

    /// <summary>Partial and pending downloads in the model store.</summary>
    public const string Downloads = "downloads";

    /// <summary>The compilation cache: compiled kernels, CUDA PTX and cubin, SPIR-V device stages.</summary>
    public const string KernelCache = "kernel-cache";

    /// <summary>Extracted shader sources.</summary>
    public const string ShaderCache = "shader-cache";

    /// <summary>Other cache files (media caches, decoded audio, thumbnails).</summary>
    public const string MediaCache = "media-cache";

    /// <summary>Log files.</summary>
    public const string Logs = "logs";

    /// <summary>Generated images, audio and video, run histories and presets.</summary>
    public const string Outputs = "outputs";

    /// <summary>Everything else Studio keeps in this location (database, settings, feed cache).</summary>
    public const string Other = "other";
}
