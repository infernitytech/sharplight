using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>The load panel: device, context and batching choices for one library model.</summary>
internal sealed partial class LoadOptionsViewModel : ObservableObject
{
    private readonly StudioServices _services;
    private readonly Action<ILoadedModel> _loaded;
    private readonly Action _closed;
    private CancellationTokenSource? _loading;

    public LoadOptionsViewModel(LibraryEntryViewModel entry, StudioServices services, Action<ILoadedModel> loaded, Action closed)
    {
        Entry = entry;
        _services = services;
        _loaded = loaded;
        _closed = closed;
        var defaults = services.Settings.Current.DefaultLoad;
        Devices = services.Runtime.Devices;
        SelectedDevice = Devices.FirstOrDefault(device => device.Id == defaults.DeviceId) ?? Devices.FirstOrDefault();
        ModelId = UniqueId(entry.Id);
        int trained = entry.Entry.TrainedContextLength ?? int.MaxValue;
        ContextLength = Math.Min(defaults.ContextLength, trained);
        BatchTokens = defaults.BatchTokens;
        KvCache = KvCacheChoice.All.FirstOrDefault(choice => choice.Value == defaults.KvCacheType) ?? KvCacheChoice.All[0];
        Slots = defaults.Slots;
        CpuOffload = defaults.CpuOffload;
        NgramDraftTokens = defaults.NgramDraftTokens;
        AllowReducedPrecision = defaults.AllowReducedPrecision;
        MtpDraftTokens = defaults.MtpDraftTokens;
    }

    /// <summary>The model to load.</summary>
    public LibraryEntryViewModel Entry { get; }

    /// <summary>Devices.</summary>
    public IReadOnlyList<ComputeDeviceOption> Devices { get; }

    /// <summary>KV cache types.</summary>
    public IReadOnlyList<KvCacheChoice> KvCacheChoices => KvCacheChoice.All;

    /// <summary>Device to load on.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanOffload))]
    public partial ComputeDeviceOption? SelectedDevice { get; set; }

    /// <summary>The id the model will be known by.</summary>
    [ObservableProperty]
    public partial string ModelId { get; set; }

    /// <summary>Context length.</summary>
    [ObservableProperty]
    public partial double ContextLength { get; set; }

    /// <summary>Rows per forward pass.</summary>
    [ObservableProperty]
    public partial double BatchTokens { get; set; }

    /// <summary>KV cache type.</summary>
    [ObservableProperty]
    public partial KvCacheChoice KvCache { get; set; }

    /// <summary>Concurrent sequences.</summary>
    [ObservableProperty]
    public partial double Slots { get; set; }

    /// <summary>Offload what does not fit to the CPU.</summary>
    [ObservableProperty]
    public partial bool CpuOffload { get; set; }

    /// <summary>Prompt-lookup draft tokens (0 = off).</summary>
    [ObservableProperty]
    public partial double NgramDraftTokens { get; set; }

    /// <summary>Multi-token-prediction draft tokens (0 = off; only models with a draft head).</summary>
    [ObservableProperty]
    public partial double MtpDraftTokens { get; set; }

    /// <summary>Overall load progress, 0 to 1.</summary>
    [ObservableProperty]
    public partial double Progress { get; set; }

    /// <summary>Opt-in to faster, less precise kernels.</summary>
    [ObservableProperty]
    public partial bool AllowReducedPrecision { get; set; }

    /// <summary>Loading.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(LoadCommand))]
    public partial bool IsLoading { get; set; }

    /// <summary>Progress or error.</summary>
    [ObservableProperty]
    public partial string Status { get; set; } = "";

    /// <summary>The trained context, as a hint.</summary>
    public string ContextHint => Entry.Entry.TrainedContextLength is { } trained ? $"Trained for {trained:N0} tokens" : "Context length per sequence";

    /// <summary>Offloading applies to GPUs only.</summary>
    public bool CanOffload => SelectedDevice?.IsGpu == true;

    /// <summary>The request the panel describes.</summary>
    internal ModelLoadRequest ToRequest()
    {
        var settings = new LoadSettings
        {
            DeviceId = SelectedDevice?.Id,
            ContextLength = Math.Max(256, (int)ContextLength),
            BatchTokens = Math.Max(16, (int)BatchTokens),
            KvCacheType = KvCache.Value,
            Slots = Math.Max(1, (int)Slots),
            CpuOffload = CpuOffload && CanOffload,
            NgramDraftTokens = Math.Max(0, (int)NgramDraftTokens),
            AllowReducedPrecision = AllowReducedPrecision,
            MtpDraftTokens = Math.Max(0, (int)MtpDraftTokens),
        };
        return new ModelLoadRequest(ModelId.Trim(), Entry.Entry.Path, settings, _services.Settings.Current.DisabledFeatures,
            Capabilities: Entry.Entry.Capabilities);
    }

    private bool CanLoad() => !IsLoading;

    [RelayCommand(CanExecute = nameof(CanLoad))]
    private async Task LoadAsync()
    {
        if (ModelId.Trim().Length == 0)
        {
            Status = "Give the model a name.";
            return;
        }

        IsLoading = true;
        Status = "Loading…";
        using var cancellation = new CancellationTokenSource();
        _loading = cancellation;
        try
        {
            Progress = 0;
            var log = new Progress<LoadStatus>(status =>
            {
                Status = status.Text;
                Progress = status.Fraction;
            });
            var model = await _services.Runtime.LoadAsync(ToRequest(), log, cancellation.Token);
            _loaded(model);
        }
        catch (OperationCanceledException)
        {
            Status = "Cancelled.";
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            Status = error.GetBaseException().Message;
        }
        finally
        {
            _loading = null;
            IsLoading = false;
        }
    }

    [RelayCommand]
    private void Close()
    {
        _loading?.Cancel();
        _closed();
    }

    private string UniqueId(string id)
    {
        var taken = _services.Runtime.LoadedModels.Select(model => model.Id).ToHashSet();
        if (!taken.Contains(id))
        {
            return id;
        }

        int suffix = 2;
        while (taken.Contains($"{id}-{suffix}"))
        {
            suffix++;
        }

        return $"{id}-{suffix}";
    }
}
