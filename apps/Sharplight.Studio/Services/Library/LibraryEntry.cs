using Sharplight.Hub;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Library;

/// <summary>A model on disk: pulled into the store, or a file the user added.</summary>
internal sealed record LibraryEntry
{
    /// <summary>The id used when loading it (the store's short name, or the file name).</summary>
    public required string Id { get; init; }

    /// <summary>Full name (<c>owner/repo:tag</c>) or file name.</summary>
    public required string DisplayName { get; init; }

    /// <summary>File or directory to load.</summary>
    public required string Path { get; init; }

    /// <summary>Architecture (e.g. <c>qwen3</c>), when known.</summary>
    public string? Architecture { get; init; }

    /// <summary>Quantization or variant tag, when known.</summary>
    public string? Quantization { get; init; }

    /// <summary>Source format.</summary>
    public string Format { get; init; } = "";

    /// <summary>Size on disk.</summary>
    public long SizeBytes { get; init; }

    /// <summary>What it can do (from the package's graphs and chat template).</summary>
    public ModelCapabilities Capabilities { get; init; }

    /// <summary>Context length the model was trained for, when known.</summary>
    public int? TrainedContextLength { get; init; }

    /// <summary>When it was pulled or added.</summary>
    public DateTimeOffset? AddedAt { get; init; }

    /// <summary>Kernel sets compiled into the package.</summary>
    public IReadOnlyList<string> Precompiled { get; init; } = [];

    /// <summary>The store record, or null for a file outside the store.</summary>
    public ModelManifest? Manifest { get; init; }

    /// <summary>True for a file outside the store.</summary>
    public bool IsLocalFile => Manifest is null;
}
