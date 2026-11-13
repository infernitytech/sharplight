namespace Sharplight.Hub;

/// <summary>How <see cref="ModelPuller"/> pulls a model.</summary>
public sealed class PullOptions
{
    /// <summary>The revision to pull (branch, tag or commit); it is resolved to a commit and every file is pinned to it.</summary>
    public string Revision { get; init; } = "main";

    /// <summary>Keep the downloaded source files in <c>source/</c> next to the package instead of deleting them.</summary>
    public bool KeepSource { get; init; }

    /// <summary>Convert again even when the stored model is already at the resolved commit.</summary>
    public bool Force { get; init; }

    /// <summary>The device to precompile the package for, or <see langword="null"/> to compile at the first load instead.</summary>
    public PrecompileTarget? Precompile { get; init; }

    /// <summary>Receives stage messages (resolving, downloading, converting, precompiling), or <see langword="null"/>.</summary>
    public Action<string>? Log { get; init; }

    /// <summary>Receives download progress over all files of the pull, or <see langword="null"/>.</summary>
    public IProgress<DownloadProgress>? Progress { get; init; }
}
