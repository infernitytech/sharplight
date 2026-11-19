using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Maintenance;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A model in the library.</summary>
internal sealed partial class LibraryEntryViewModel : ObservableObject
{
    public LibraryEntryViewModel(LibraryEntry entry)
    {
        Entry = entry;
        Badges = new CapabilityBadges(entry.Capabilities);
        KindInfo = ModelKindCatalog.For(entry.Kind);
    }

    /// <summary>How the model's kind is presented (name, icon, whether the engine runs it).</summary>
    public ModelKindInfo KindInfo { get; }

    /// <summary>The kind's name (<c>Text generation</c>).</summary>
    public string KindTitle => KindInfo.Title;

    /// <summary>The kind's icon resource key.</summary>
    public string KindIcon => KindInfo.Icon;

    /// <summary>Tooltip of the kind badge: whether it is a guess and whether the engine runs it.</summary>
    public string KindToolTip => KindInfo.Id == ModelKindCatalog.Lora
        ? "LoRA adapter: choose it in the load panel of an image model"
        : KindInfo.Runnable
        ? KindInfo.Title + (Entry.KindInferred ? " (guessed from the file type)" : "")
        : KindInfo.Title + ": Sharplight cannot run models of this kind yet" + (Entry.KindInferred ? " (guessed from the file type)" : "");

    /// <summary>The engine can load and run this kind.</summary>
    public bool IsRunnable => KindInfo.Runnable;

    /// <summary>The model; replaced when it is read again from disk (after precompiling).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Size), nameof(SizeBytes), nameof(IsPrecompiled), nameof(PrecompiledTip), nameof(LastUsedOrAdded))]
    public partial LibraryEntry Entry { get; set; }

    /// <summary>The library listing the entry, for the commands of its context menu.</summary>
    public LibraryViewModel? Library { get; init; }

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

    /// <summary>The precompiled sets, one per line, for the badge's tooltip.</summary>
    public string PrecompiledTip => "Precompiled code stored in the package (loads fast):\n" + string.Join("\n", Entry.Precompiled);

    /// <summary>The model came from the Sharplight model feed (a signed, preconverted package).</summary>
    public bool IsFromFeed => Entry.Manifest?.FeedPackage is not null;

    /// <summary>The installed feed version, e.g. <c>Feed 1.0.0</c>.</summary>
    public string FeedText => Entry.Manifest?.FeedVersion is { } version ? $"Feed {version}" : "";

    /// <summary>A newer version than the installed one is in the feed; null when none is known.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasFeedUpdate), nameof(FeedUpdateText), nameof(CanUpdateFromFeed))]
    public partial string? FeedUpdateVersion { get; set; }

    /// <summary>The model is being replaced by the newer version.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanUpdateFromFeed))]
    public partial bool IsUpdating { get; set; }

    /// <summary>A newer feed version exists.</summary>
    public bool HasFeedUpdate => FeedUpdateVersion is not null;

    /// <summary>The badge text, e.g. <c>Update 1.1.0</c>.</summary>
    public string FeedUpdateText => FeedUpdateVersion is { } version ? $"Update {version}" : "";

    /// <summary>The update button is offered.</summary>
    public bool CanUpdateFromFeed => HasFeedUpdate && !IsUpdating;

    /// <summary>A file outside the store.</summary>
    public bool IsLocalFile => Entry.IsLocalFile;

    /// <summary>A <c>.slm</c> package file, which the package inspector can open and the precompiler update.</summary>
    public bool IsPackage => Entry.Path.EndsWith(ModelPackage.Extension, StringComparison.OrdinalIgnoreCase) && File.Exists(Entry.Path);

    /// <summary>Loaded right now (registered with the runtime, which maps the file).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanPrecompile), nameof(PrecompileHint))]
    public partial bool IsLoaded { get; set; }


    /// <summary>Chosen in the selection mode for a bulk action.</summary>
    [ObservableProperty]
    public partial bool IsSelected { get; set; }

    /// <summary>When the model was last loaded, or null when it was not since tracking began.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(LastUsedText), nameof(LastUsedOrAdded))]
    public partial DateTimeOffset? LastUsed { get; set; }

    /// <summary>When it was last loaded; for a model never loaded, when it was added (so a new download is not "unused").</summary>
    public DateTimeOffset? LastUsedOrAdded => LastUsed ?? Entry.AddedAt;

    /// <summary><c>Used 3 days ago</c>, or <c>Never loaded</c>.</summary>
    public string LastUsedText => LastUsed is { } used ? $"Used {RelativeTime.Format(used, DateTimeOffset.UtcNow)}" : "Never loaded";

    /// <summary>The size in bytes.</summary>
    public long SizeBytes => Entry.SizeBytes;

    /// <summary>Where the bytes are, once read (the size details of the card).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasBreakdown))]
    public partial ModelSizeBreakdown? Breakdown { get; set; }

    /// <summary>The breakdown is shown.</summary>
    [ObservableProperty]
    public partial bool IsBreakdownOpen { get; set; }

    /// <summary>The breakdown is being read.</summary>
    [ObservableProperty]
    public partial bool IsLoadingBreakdown { get; set; }

    /// <summary>The breakdown was read.</summary>
    public bool HasBreakdown => Breakdown is not null;

    /// <summary>The lines of <see cref="Breakdown"/>: weights, attachments, precompiled artifacts per provider and device, the rest.</summary>
    public ObservableCollection<BreakdownLineViewModel> BreakdownLines { get; } = [];

    /// <summary>The model's package is exported with or without artifacts, imported or deleted right now.</summary>
    [ObservableProperty]
    public partial bool IsBusy { get; set; }

    /// <summary>Fills <see cref="BreakdownLines"/> from <see cref="Breakdown"/>.</summary>
    public void ShowBreakdown(ModelSizeBreakdown breakdown)
    {
        Breakdown = breakdown;
        BreakdownLines.Clear();
        if (!breakdown.IsPackage)
        {
            BreakdownLines.Add(new BreakdownLineViewModel("Files", "not a .slm package: only the total is known", breakdown.TotalBytes, breakdown.TotalBytes));
            return;
        }

        BreakdownLines.Add(new BreakdownLineViewModel("Weights", "", breakdown.WeightBytes, breakdown.TotalBytes));
        if (breakdown.AttachmentBytes > 0)
        {
            BreakdownLines.Add(new BreakdownLineViewModel("Attachments", "tokenizer, scheduler and other files", breakdown.AttachmentBytes, breakdown.TotalBytes));
        }

        foreach (var artifact in breakdown.Artifacts)
        {
            string title = artifact.Device.Length == 0 ? $"Precompiled · {artifact.Provider}" : $"Precompiled · {artifact.Provider} · {artifact.Device}";
            string detail = $"{ByteFormat.Count(artifact.Entries, "entry", "entries")}{(artifact.Description.Length == 0 ? "" : " · " + artifact.Description)}";
            BreakdownLines.Add(new BreakdownLineViewModel(title, detail, artifact.Bytes, breakdown.TotalBytes));
        }

        if (breakdown.ReclaimableBytes > 0)
        {
            BreakdownLines.Add(new BreakdownLineViewModel("Unused space", "replaced artifacts, removed by compacting", breakdown.ReclaimableBytes, breakdown.TotalBytes));
        }

        BreakdownLines.Add(new BreakdownLineViewModel("Definition and index", "graphs, weight index, padding", breakdown.OtherBytes, breakdown.TotalBytes));
    }

    /// <summary>Being precompiled right now.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanPrecompile), nameof(PrecompileHint), nameof(HasPrecompileDetails))]
    public partial bool IsPrecompiling { get; set; }

    /// <summary>The last precompile's summary or error, or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasPrecompileDetails))]
    public partial string? PrecompileStatus { get; set; }

    /// <summary>Progress and results per provider and device of the running or last precompile.</summary>
    public ObservableCollection<PrecompileRowViewModel> PrecompileRows { get; } = [];

    /// <summary>Show the precompile panel of the card.</summary>
    public bool HasPrecompileDetails => IsPrecompiling || PrecompileStatus is not null;

    /// <summary>
    /// The Precompile action is available: a package that is neither resident (the package is rewritten in place, so it must not be
    /// mapped by an engine) nor being precompiled.
    /// </summary>
    public bool CanPrecompile => IsPackage && !IsLoaded && !IsPrecompiling;

    /// <summary>What the Precompile action does, or why it is unavailable.</summary>
    public string PrecompileHint => IsPrecompiling ? "Precompiling…"
        : !IsPackage ? "Only .slm packages can be precompiled: convert the model first."
        : IsLoaded ? "Unload the model to precompile it: the package is rewritten in place and is in use while the model is resident."
        : "Compile the GPU code for every execution provider and device on this machine and store it in the package, so loads skip compilation.";

    /// <summary>Cancels the running precompile.</summary>
    internal CancellationTokenSource? PrecompileCancellation { get; set; }

    /// <summary>Starts showing a precompile run.</summary>
    public void BeginPrecompile()
    {
        PrecompileRows.Clear();
        PrecompileStatus = null;
        IsPrecompiling = true;
    }

    /// <summary>Applies a progress report (on the UI thread).</summary>
    public void OnPrecompileProgress(PrecompileProgress update)
    {
        if (update.Provider.Length == 0)
        {
            PrecompileStatus = update.Message;
            return;
        }

        Row(update.Provider, update.Device).Apply(update);
    }

    /// <summary>Shows the result of a finished precompile: per device, skipped providers, and a summary line.</summary>
    public void CompletePrecompile(PackagePrecompileReport report)
    {
        foreach (var device in report.Devices)
        {
            Row(device.Provider, device.Device).Complete(device);
        }

        foreach (var skipped in report.Skipped)
        {
            Row(skipped.Provider, skipped.Device).Skip(skipped.Reason);
        }

        PrecompileStatus = Summarize(report);
    }

    /// <summary><c>Stored 3 artifact sets (14.2 MB) for 3 devices in 41 s · skipped: cpu (nothing to precompile)</c>.</summary>
    public static string Summarize(PackagePrecompileReport report)
    {
        int devices = report.Devices.Count(d => d.Portable is not null || d.DeviceCode is not null);
        var stored = report.StoredSets.Count == 0
            ? $"Already up to date for {ByteFormat.Count(devices, "device", "devices")}"
            : $"Stored {ByteFormat.Count(report.StoredSets.Count, "artifact set", "artifact sets")} ({ByteFormat.Format(report.StoredBytes)}) for {ByteFormat.Count(devices, "device", "devices")}";
        var skipped = report.Devices.Where(d => d.Portable is null && d.DeviceCode is null).Select(d => $"{d.Provider} ({d.Note ?? "nothing to precompile"})")
            .Concat(report.Skipped.Select(s => $"{s.Provider}{(s.Device.Length == 0 ? "" : " " + s.Device)} ({s.Reason})")).ToList();
        var text = $"{stored} in {ByteFormat.Duration(report.Elapsed)}";
        return skipped.Count == 0 ? text : $"{text} · skipped: {string.Join(", ", skipped)}";
    }

    private PrecompileRowViewModel Row(string provider, string device)
    {
        var row = PrecompileRows.FirstOrDefault(r => r.Provider == provider && r.Device == device);
        if (row is null)
        {
            row = new PrecompileRowViewModel(provider, device);
            PrecompileRows.Add(row);
        }

        return row;
    }
}
