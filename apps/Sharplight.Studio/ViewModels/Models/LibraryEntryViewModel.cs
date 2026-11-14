using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A model in the library.</summary>
internal sealed partial class LibraryEntryViewModel : ObservableObject
{
    public LibraryEntryViewModel(LibraryEntry entry)
    {
        Entry = entry;
        Badges = new CapabilityBadges(entry.Capabilities);
    }

    /// <summary>The model.</summary>
    public LibraryEntry Entry { get; }

    /// <summary>Id used when loading.</summary>
    public string Id => Entry.Id;

    /// <summary>Full name.</summary>
    public string Name => Entry.DisplayName;

    /// <summary><c>qwen3 · Q4_K_M · GGUF</c>.</summary>
    public string Detail => string.Join(" · ", new[] { Entry.Architecture, Entry.Quantization, Entry.Format }.Where(part => !string.IsNullOrEmpty(part)));

    /// <summary>Size on disk.</summary>
    public string Size => ByteFormat.Format(Entry.SizeBytes);

    /// <summary>Trained context, e.g. <c>40.9k ctx</c>.</summary>
    public string? Context => Entry.TrainedContextLength is { } context ? $"{ByteFormat.Count(context)} ctx" : null;

    /// <summary>When it was added.</summary>
    public string? Added => Entry.AddedAt is { } added ? RelativeTime.Format(added, DateTimeOffset.UtcNow) : null;

    /// <summary>Capability badges.</summary>
    public CapabilityBadges Badges { get; }

    /// <summary>Kernels are precompiled into the package.</summary>
    public bool IsPrecompiled => Entry.Precompiled.Count > 0;

    /// <summary>A file outside the store.</summary>
    public bool IsLocalFile => Entry.IsLocalFile;

    /// <summary>Loaded right now.</summary>
    [ObservableProperty]
    public partial bool IsLoaded { get; set; }

    /// <summary>The delete button was pressed once (press again to confirm).</summary>
    [ObservableProperty]
    public partial bool ConfirmDelete { get; set; }
}
