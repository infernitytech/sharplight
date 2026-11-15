using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Compatibility;
using Sharplight.Studio.Services.Fit;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>
/// The load panel: device, context and batching choices for one library model, with an estimate of whether it fits the device's
/// memory that follows every change of the choices.
/// </summary>
internal sealed partial class LoadOptionsViewModel : ObservableObject
{
    /// <summary>Choices whose change re-estimates the fit.</summary>
    private static readonly HashSet<string> s_estimated = new(StringComparer.Ordinal)
    {
        nameof(SelectedDevice), nameof(ContextLength), nameof(BatchTokens), nameof(KvCache), nameof(Slots), nameof(CpuOffload),
        nameof(MtpDraftTokens), nameof(AllowReducedPrecision),
    };

    private readonly StudioServices _services;
    private readonly Action<ILoadedModel> _loaded;
    private readonly Action _closed;
    private readonly Func<IReadOnlyList<LibraryEntry>> _library;
    private readonly Action<LibraryEntry>? _openVariant;
    private readonly FitAdvisor _advisor;
    private readonly EventHandler _modelsChanged;
    private CancellationTokenSource? _loading;
    private CancellationTokenSource? _estimating;
    private bool _ready;

    /// <param name="entry">The model to load.</param>
    /// <param name="services">The services.</param>
    /// <param name="loaded">Called with the model once it loaded.</param>
    /// <param name="closed">Called when the panel closes.</param>
    /// <param name="library">The library, for smaller quantizations of the same model to suggest.</param>
    /// <param name="openVariant">Opens the load panel for a suggested smaller quantization.</param>
    public LoadOptionsViewModel(LibraryEntryViewModel entry, StudioServices services, Action<ILoadedModel> loaded, Action closed,
        Func<IReadOnlyList<LibraryEntry>>? library = null, Action<LibraryEntry>? openVariant = null)
    {
        Entry = entry;
        _services = services;
        _loaded = loaded;
        _closed = closed;
        _library = library ?? (() => []);
        _openVariant = openVariant;
        _advisor = new FitAdvisor(services.Runtime, services.Settings);
        _modelsChanged = (_, _) => services.Dispatcher.Post(() =>
        {
            if (!IsLoading)
            {
                ScheduleEstimate();
            }
        });
        var defaults = services.Settings.Current.DefaultLoad;
        Devices = services.Runtime.Devices;
        SelectedDevice = Devices.FirstOrDefault(device => device.Id == defaults.DeviceId) ?? Devices.FirstOrDefault();
        ModelId = UniqueId(entry.Id);
        NeedsCompanions = entry.Entry.Kind == Sharplight.Models.ModelKinds.ImageGeneration && File.Exists(entry.Entry.Path)
            && ImageModelProbe.NeedsCompanions(entry.Entry.Path);
        if (NeedsCompanions)
        {
            var found = services.Runtime.FindCompanions(entry.Entry.Path);
            TextEncoderPath = found.GetValueOrDefault(AuxiliaryRoles.TextEncoder, "");
            SecondTextEncoderPath = found.GetValueOrDefault(AuxiliaryRoles.SecondTextEncoder, "");
            VaePath = found.GetValueOrDefault(AuxiliaryRoles.Vae, "");
        }

        int trained = entry.Entry.TrainedContextLength ?? int.MaxValue;
        ContextLength = Math.Min(defaults.ContextLength, trained);
        BatchTokens = defaults.BatchTokens;
        KvCache = KvCacheChoice.All.FirstOrDefault(choice => choice.Value == defaults.KvCacheType) ?? KvCacheChoice.All[0];
        Slots = defaults.Slots;
        CpuOffload = defaults.CpuOffload;
        NgramDraftTokens = defaults.NgramDraftTokens;
        AllowReducedPrecision = defaults.AllowReducedPrecision;
        MtpDraftTokens = defaults.MtpDraftTokens;
        _ready = true;
        services.Runtime.ModelsChanged += _modelsChanged;
        ScheduleEstimate();
    }

    /// <summary>How long the choices must stay unchanged before they are estimated.</summary>
    internal static TimeSpan EstimateDelay { get; set; } = TimeSpan.FromMilliseconds(250);

    /// <summary>The model to load.</summary>
    public LibraryEntryViewModel Entry { get; }

    /// <summary>The model generates text, so context, batching, KV cache and speculation apply (the other kinds load with defaults).</summary>
    public bool IsLanguageModel => !TaskKindClients.Handles(Entry.Entry.Kind);

    /// <summary>A speech synthesis model given as a language-model file needs the codec decoder file next to it.</summary>
    public bool NeedsVocoder => Entry.Entry.Kind == Sharplight.Models.ModelKinds.SpeechSynthesis
        && !Entry.Entry.Path.EndsWith(".slm", StringComparison.OrdinalIgnoreCase);

    /// <summary>The codec decoder file of a speech synthesis model (<see cref="NeedsVocoder"/>).</summary>
    [ObservableProperty]
    public partial string VocoderPath { get; set; } = "";

    /// <summary>
    /// An image model file that holds only the diffusion transformer (a FLUX GGUF) needs its text encoders and VAE as companion files;
    /// the ones found beside it are filled in.
    /// </summary>
    public bool NeedsCompanions { get; }

    /// <summary>The CLIP text encoder file of a transformer-only image model (<see cref="NeedsCompanions"/>).</summary>
    [ObservableProperty]
    public partial string TextEncoderPath { get; set; } = "";

    /// <summary>The T5 text encoder file of a transformer-only image model (<see cref="NeedsCompanions"/>).</summary>
    [ObservableProperty]
    public partial string SecondTextEncoderPath { get; set; } = "";

    /// <summary>The VAE file of a transformer-only image model (<see cref="NeedsCompanions"/>).</summary>
    [ObservableProperty]
    public partial string VaePath { get; set; } = "";

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

    /// <summary>The memory estimate of the current choices, once computed.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasFit))]
    public partial FitEstimateViewModel? Fit { get; set; }

    /// <summary>An estimate is shown.</summary>
    public bool HasFit => Fit is not null;

    /// <summary>The choices are being estimated.</summary>
    [ObservableProperty]
    public partial bool IsEstimating { get; set; }

    /// <summary>Why there is no estimate (the model could not be read), else empty.</summary>
    [ObservableProperty]
    public partial string FitError { get; set; } = "";

    /// <summary>The latest estimate (for tests).</summary>
    internal Task Estimate { get; private set; } = Task.CompletedTask;

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
        var auxiliary = new List<(string Role, string Path)>();
        if (NeedsVocoder && VocoderPath.Trim().Length > 0)
        {
            auxiliary.Add((AuxiliaryRoles.Vocoder, VocoderPath.Trim()));
        }

        if (NeedsCompanions)
        {
            foreach (var (role, path) in Companions())
            {
                auxiliary.Add((role, path));
            }
        }

        return new ModelLoadRequest(ModelId.Trim(), Entry.Entry.Path, settings, _services.Settings.Current.DisabledFeatures,
            Capabilities: Entry.Entry.Capabilities, Kind: Entry.Entry.Kind, Auxiliary: auxiliary.Count > 0 ? auxiliary : null);
    }

    /// <summary>The companion files given, by auxiliary role.</summary>
    private IEnumerable<(string Role, string Path)> Companions()
    {
        (string Role, string Path)[] all =
        [
            (AuxiliaryRoles.TextEncoder, TextEncoderPath.Trim()),
            (AuxiliaryRoles.SecondTextEncoder, SecondTextEncoderPath.Trim()),
            (AuxiliaryRoles.Vae, VaePath.Trim()),
        ];
        return all.Where(companion => companion.Path.Length > 0);
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

        if (NeedsVocoder && !File.Exists(VocoderPath.Trim()))
        {
            Status = "Choose the vocoder file (the WavTokenizer .gguf that goes with this model).";
            return;
        }

        if (NeedsCompanions && MissingCompanion() is { } missing)
        {
            Status = $"Choose the {missing} file: this model file holds only the diffusion transformer.";
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
            Detach();
            _loaded(model);
        }
        catch (OperationCanceledException)
        {
            Status = "Cancelled.";
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            Status = ImportFailure.Describe(error, error.GetBaseException().Message, Entry.Entry.Path);
        }
        finally
        {
            _loading = null;
            IsLoading = false;
        }
    }

    [RelayCommand]
    private async Task BrowseVocoderAsync()
    {
        var files = await _services.Platform.PickFilesAsync("Choose the vocoder", [new FileFilter("GGUF files", ["*.gguf"])], allowMultiple: false);
        if (files.Count > 0)
        {
            VocoderPath = files[0];
        }
    }

    [RelayCommand]
    private async Task BrowseTextEncoderAsync() => TextEncoderPath = await PickCompanionAsync("Choose the CLIP-L text encoder", TextEncoderPath);

    [RelayCommand]
    private async Task BrowseSecondTextEncoderAsync() => SecondTextEncoderPath = await PickCompanionAsync("Choose the T5-XXL text encoder", SecondTextEncoderPath);

    [RelayCommand]
    private async Task BrowseVaeAsync() => VaePath = await PickCompanionAsync("Choose the VAE", VaePath);

    private async Task<string> PickCompanionAsync(string title, string current)
    {
        var files = await _services.Platform.PickFilesAsync(title, [new FileFilter("Model files", ["*.safetensors", "*.gguf"])], allowMultiple: false);
        return files.Count > 0 ? files[0] : current;
    }

    /// <summary>The first companion that is not an existing file, as the user knows it, or null.</summary>
    private string? MissingCompanion()
        => !File.Exists(TextEncoderPath.Trim()) ? "CLIP-L text encoder"
            : !File.Exists(SecondTextEncoderPath.Trim()) ? "T5-XXL text encoder"
            : !File.Exists(VaePath.Trim()) ? "VAE"
            : null;

    [RelayCommand]
    private void Close()
    {
        _loading?.Cancel();
        Detach();
        _closed();
    }

    /// <inheritdoc/>
    protected override void OnPropertyChanged(PropertyChangedEventArgs e)
    {
        base.OnPropertyChanged(e);
        if (_ready && e.PropertyName is { } name && s_estimated.Contains(name))
        {
            ScheduleEstimate();
        }
    }

    /// <summary>Stops following the runtime and cancels an estimate in progress.</summary>
    private void Detach()
    {
        _services.Runtime.ModelsChanged -= _modelsChanged;
        _estimating?.Cancel();
    }

    /// <summary>Estimates the current choices once they stay unchanged for <see cref="EstimateDelay"/>.</summary>
    private void ScheduleEstimate()
    {
        _estimating?.Cancel();
        var cancellation = new CancellationTokenSource();
        _estimating = cancellation;
        Estimate = EstimateAsync(cancellation);
    }

    private async Task EstimateAsync(CancellationTokenSource cancellation)
    {
        var token = cancellation.Token;
        if (!IsLanguageModel)
        {
            // The engine's footprint estimate covers language models; the load checks the memory itself.
            return;
        }

        try
        {
            await Task.Delay(EstimateDelay, token);
            IsEstimating = true;
            var request = ToRequest();
            var memory = await _advisor.GetMemoryAsync(token);
            var footprint = await _services.Runtime.EstimateFootprintAsync(request, token);
            var report = _advisor.Plan(footprint, request.Settings, memory);
            token.ThrowIfCancellationRequested();
            FitError = "";
            Fit = new FitEstimateViewModel(report, []);
            var suggestions = await _advisor.SuggestAsync(request, report, memory, Entry.Entry, _library(), token);
            token.ThrowIfCancellationRequested();
            if (suggestions.Count > 0)
            {
                Fit = new FitEstimateViewModel(report, [.. suggestions.Select(suggestion => new FitSuggestionViewModel(suggestion, Apply))]);
            }
        }
        catch (OperationCanceledException)
        {
            // Superseded by a newer estimate or the panel closed.
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            if (!token.IsCancellationRequested)
            {
                Fit = null;
                FitError = $"No memory estimate: {error.GetBaseException().Message}";
            }
        }
        finally
        {
            if (_estimating == cancellation)
            {
                IsEstimating = false;
            }
        }
    }

    /// <summary>Applies a suggested change to the choices (or opens the suggested smaller quantization).</summary>
    private void Apply(FitSuggestion suggestion)
    {
        if (suggestion.Variant is { } variant)
        {
            _openVariant?.Invoke(variant);
            return;
        }

        if (suggestion.Settings is not { } settings)
        {
            return;
        }

        SelectedDevice = Devices.FirstOrDefault(device => device.Id == settings.DeviceId) ?? SelectedDevice;
        ContextLength = settings.ContextLength;
        KvCache = KvCacheChoice.All.FirstOrDefault(choice => choice.Value == settings.KvCacheType) ?? KvCache;
        Slots = settings.Slots;
        CpuOffload = settings.CpuOffload && CanOffload;
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
