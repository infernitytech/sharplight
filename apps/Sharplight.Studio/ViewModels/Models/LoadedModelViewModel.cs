using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A resident model: state, memory, load progress and settings.</summary>
internal sealed partial class LoadedModelViewModel : ObservableObject
{
    public LoadedModelViewModel(ILoadedModel model)
    {
        Model = model;
        Refresh();
    }

    /// <summary>The model.</summary>
    public ILoadedModel Model { get; }

    /// <summary>Its id.</summary>
    public string Id => Model.Id;

    /// <summary>The text of the button that opens the model's page: <c>Chat</c>, <c>Transcribe</c>, <c>Speak</c>, <c>Image</c>, <c>Vision</c>, <c>Audio</c>.</summary>
    public string OpenText => Model.Kind switch
    {
        Sharplight.Models.ModelKinds.SpeechRecognition => "Transcribe",
        Sharplight.Models.ModelKinds.SpeechSynthesis => "Speak",
        Sharplight.Models.ModelKinds.ImageGeneration => "Image",
        Sharplight.Models.ModelKinds.AudioCodec or Sharplight.Models.ModelKinds.VoiceActivity => "Audio",
        Sharplight.Models.ModelKinds.ImageClassification or Sharplight.Models.ModelKinds.ImageEmbedding or Sharplight.Models.ModelKinds.ObjectDetection
            or Sharplight.Models.ModelKinds.DepthEstimation or Sharplight.Models.ModelKinds.ImageSegmentation or Sharplight.Models.ModelKinds.Ocr
            or Sharplight.Models.ModelKinds.ImageUpscaling => "Vision",
        _ => "Chat",
    };

    /// <summary>The icon resource key of the button that opens the model's page.</summary>
    public string OpenIcon => ModelKindCatalog.For(Model.Kind).Icon;

    /// <summary>Device name.</summary>
    public string Device => Model.Device.Name;

    /// <summary>Life-cycle state.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsLoading), nameof(IsReady), nameof(IsBusy), nameof(IsSleeping), nameof(IsFailed))]
    public partial ResidentState State { get; set; }

    /// <summary>State as text, e.g. <c>Ready</c>, <c>Generating (2)</c>.</summary>
    [ObservableProperty]
    public partial string StateText { get; set; } = "";

    /// <summary>Memory: resident while loaded, the estimate otherwise.</summary>
    [ObservableProperty]
    public partial string MemoryText { get; set; } = "";

    /// <summary>Load progress, 0..1.</summary>
    [ObservableProperty]
    public partial double LoadProgress { get; set; }

    /// <summary>Load stages, e.g. <c>Compiling kernels 40 / 107</c>.</summary>
    [ObservableProperty]
    public partial string LoadText { get; set; } = "";

    /// <summary>Last use, relative.</summary>
    [ObservableProperty]
    public partial string LastUsedText { get; set; } = "";

    /// <summary>Settings summary.</summary>
    [ObservableProperty]
    public partial string Detail { get; set; } = "";

    /// <summary>Capability badges.</summary>
    [ObservableProperty]
    public partial CapabilityBadges Badges { get; set; } = new(ModelCapabilities.None);

    /// <summary>Unloading was asked for (the button shows progress).</summary>
    [ObservableProperty]
    public partial bool IsUnloading { get; set; }

    /// <summary>Loading.</summary>
    public bool IsLoading => State == ResidentState.Loading;

    /// <summary>Loaded and idle.</summary>
    public bool IsReady => State == ResidentState.Ready;

    /// <summary>Generating.</summary>
    public bool IsBusy => State == ResidentState.Busy;

    /// <summary>Unloaded for idleness or memory; reloads on use.</summary>
    public bool IsSleeping => State == ResidentState.Sleeping;

    /// <summary>The last load failed.</summary>
    public bool IsFailed => State == ResidentState.Failed;

    /// <summary>Pinned (never evicted).</summary>
    public bool IsPinned => Model.IsPinned;

    /// <summary>Re-reads the live state.</summary>
    public void Refresh()
    {
        State = Model.State;
        StateText = State switch
        {
            ResidentState.Busy => Model.ActiveRequests == 1 ? "Generating" : $"Generating ({Model.ActiveRequests})",
            ResidentState.Sleeping => "Sleeping · loads on use",
            ResidentState.Failed => "Failed: " + (Model.Error ?? "unknown error"),
            var state => state.ToString(),
        };
        MemoryText = Model.ResidentBytes > 0
            ? ByteFormat.Format(Model.ResidentBytes)
            : Model.EstimatedBytes > 0 ? "~" + ByteFormat.Format(Model.EstimatedBytes) : "";
        LoadProgress = Model.LoadStatus?.Fraction ?? 0;
        LoadText = Model.LoadStatus?.Text ?? "";
        LastUsedText = Model.LastUsed is { } used ? "used " + RelativeTime.Format(used, DateTimeOffset.UtcNow) : "";
        Badges = new CapabilityBadges(Model.Capabilities);
        Detail = BuildDetail();
    }

    private string BuildDetail()
    {
        var settings = Model.Settings;
        var parts = new List<string>
        {
            $"{ByteFormat.Count(Model.ContextLength)} ctx",
            $"{settings.Slots.ToString(CultureInfo.InvariantCulture)} slots",
            $"KV {settings.KvCacheType}",
        };
        if (settings.NgramDraftTokens > 0)
        {
            parts.Add($"lookup drafts {settings.NgramDraftTokens}");
        }

        if (settings.MtpDraftTokens > 0 && Model.Capabilities.HasFlag(ModelCapabilities.DraftHead))
        {
            parts.Add($"draft head {settings.MtpDraftTokens} tokens");
        }

        if (settings.CpuOffload)
        {
            parts.Add("CPU offload");
        }

        if (settings.AllowReducedPrecision)
        {
            parts.Add("reduced precision");
        }

        return string.Join(" · ", parts);
    }
}
