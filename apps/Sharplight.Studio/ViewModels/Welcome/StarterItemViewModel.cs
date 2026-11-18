using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.FirstRun;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.ViewModels.Welcome;

/// <summary>One recommended package in the welcome flow: what it is for, what it costs to download and whether it is ticked.</summary>
internal sealed partial class StarterItemViewModel : ObservableObject
{
    public StarterItemViewModel(StarterItem item, FeedPackageViewModel package, string timeText)
    {
        Item = item;
        Package = package;
        IsSelected = item.Preselected;
        TimeText = timeText;
    }

    /// <summary>The recommendation.</summary>
    public StarterItem Item { get; }

    /// <summary>The package as the feed list shows it (installs go through it).</summary>
    public FeedPackageViewModel Package { get; }

    /// <summary>The package is installed already, so nothing is downloaded for it.</summary>
    public bool IsInstalled => Package.IsInstalled && !Package.HasUpdate;

    /// <summary>The chat model cannot be unticked: the flow is about chatting.</summary>
    public bool IsRequired => Item.Role == StarterRole.Chat;

    /// <summary>The tick is changeable.</summary>
    public bool CanToggle => !IsRequired;

    /// <summary>What it is for.</summary>
    public string RoleTitle => Item.Role switch
    {
        StarterRole.Chat => "Chat",
        StarterRole.Speech => "Speech to text",
        _ => "Search embeddings",
    };

    /// <summary>The package's name and variant.</summary>
    public string Title => Package.Title;

    /// <summary>The package's one-sentence description.</summary>
    public string Description => Package.Description;

    /// <summary>Bytes still to download.</summary>
    public long DownloadBytes => IsInstalled ? 0 : Item.DownloadBytes;

    /// <summary>The download size, or that it is installed.</summary>
    public string SizeText => IsInstalled ? "Already installed" : ByteFormat.Format(Item.DownloadBytes);

    /// <summary>The estimated download time, empty for an installed package.</summary>
    public string TimeText { get; }

    /// <summary>Where it runs and how much memory it needs.</summary>
    public string FitText => $"{(Item.OnGpu ? "Runs on your GPU" : "Runs on the CPU")} · needs about {ByteFormat.Format(Item.MemoryBytes)}";

    [ObservableProperty]
    public partial bool IsSelected { get; set; }

    /// <summary>The step: empty, <c>Downloading…</c>, <c>Ready</c> or the failure.</summary>
    [ObservableProperty]
    public partial string State { get; set; } = "";
}
