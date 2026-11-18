using Microsoft.Extensions.Logging;

namespace Sharplight.Studio.Services.Logging;

/// <summary>Where and how much Studio's log file keeps.</summary>
internal sealed class FileLogOptions
{
    /// <summary>The log folder (<c>StorageLayout.LogsDirectory</c>, so it follows the storage mode).</summary>
    public required string Directory { get; init; }

    /// <summary>The base name of the files: <c>studio.log</c>, then <c>studio.1.log</c> (newest of the older ones) and so on.</summary>
    public string FileName { get; init; } = "studio.log";

    /// <summary>The size at which the current file is rolled over.</summary>
    public long MaxFileBytes { get; init; } = 2L * 1024 * 1024;

    /// <summary>How many files are kept in total (the current one and the rolled ones).</summary>
    public int MaxFiles { get; init; } = 5;

    /// <summary>The lowest level written.</summary>
    public LogLevel MinimumLevel { get; init; } = LogLevel.Information;
}
