using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Hosting.Fit;
using Sharplight.Hosting.Loaders;
using Sharplight.Runtime;
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
        nameof(ExpertsOnDisk), nameof(MtpDraftTokens), nameof(AllowReducedPrecision),
    };

    private readonly StudioServices _services;
    private readonly Action<ILoadedModel> _loaded;
    private readonly Action _closed;
    private readonly Func<IReadOnlyList<LibraryEntry>> _library;
    private readonly Action<LibraryEntry>? _openVariant;
    private readonly FitAdvisor _advisor;
    private readonly EventHandler _modelsChanged;
    private readonly EventHandler<IReadOnlyList<DeviceMemory>> _memoryChanged;
    private readonly IDisposable _watch;
    private readonly Lock _scheduleLock = new();
    private CancellationTokenSource? _loading;
    private CancellationTokenSource? _estimating;
    private bool _ready;
    private bool _detached;
    private IReadOnlyList<DeviceMemory>? _pendingMemory;
    private IReadOnlyList<DeviceMemory>? _plannedMemory;
    private ModelWeightFormats? _weightFormats;

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
        _advisor = FitAdvisor.For(services);
        _modelsChanged = (_, _) => services.Dispatcher.Post(() =>
        {
            if (!IsLoading)
            {
                ScheduleEstimate();
            }
        });
        // New figures from the periodic read while the panel is open (other applications, a model loading elsewhere): plan again with them.
        _memoryChanged = (_, memory) => services.Dispatcher.Post(() =>
        {
            // Figures the current plan already used (the panel's own read, shared with the watch) change nothing.
            if (IsLoading || _detached || DeviceMemoryWatch.Same(Volatile.Read(ref _plannedMemory), memory))
            {
                return;
            }

            if (Estimate.IsCompleted)
            {
                ScheduleEstimate(memory);
            }
            else
            {
                // The estimate in progress reads the same figures (one shared read) or older ones: it plans again when it is done.
                _pendingMemory = memory;
            }
        });
        var defaults = services.Settings.Current.DefaultLoad;
        Devices = services.Runtime.Devices;
        SelectedDevice = Devices.FirstOrDefault(device => device.Id == defaults.DeviceId) ?? Devices.FirstOrDefault();
        ModelId = UniqueId(entry.Id);
        Loras = new Image.LoraListViewModel(services.Platform, _library);
        NeedsCompanions = entry.Entry.Kind == Sharplight.Models.ModelKinds.ImageGeneration && File.Exists(entry.Entry.Path)
            && ImageModelProbe.NeedsCompanions(entry.Entry.Path);
        if (NeedsCompanions)
        {
            var found = services.Runtime.FindCompanions(entry.Entry.Path);
            Companions = [.. services.Runtime.DescribeCompanions(entry.Entry.Path)
                .Select(file => new CompanionFileViewModel(file.Role, file.Label, file.Example, found.GetValueOrDefault(file.Role, ""), PickCompanionAsync))];
        }

        int trained = entry.Entry.TrainedContextLength ?? int.MaxValue;
        ContextLength = Math.Min(defaults.ContextLength, trained);
        BatchTokens = defaults.BatchTokens;
        KvCache = KvCacheChoice.All.FirstOrDefault(choice => choice.Value == defaults.KvCacheType) ?? KvCacheChoice.All[0];
        Slots = defaults.Slots;
        CpuOffload = defaults.CpuOffload;
        ExpertsOnDisk = defaults.ExpertsOnDisk;
        NgramDraftTokens = defaults.NgramDraftTokens;
        AllowReducedPrecision = defaults.AllowReducedPrecision;
        MtpDraftTokens = defaults.MtpDraftTokens;
        BlockScaledFp8 = defaults.BlockScaledFp8;
        Fp4Activations = defaults.Fp4Activations;
        StepFloor = StepFloorChoice.All.FirstOrDefault(choice => choice.Value == defaults.StepFloor) ?? StepFloorChoice.All[0];
        StepRowsPerSequence = defaults.StepRowsPerSequence ?? 0;
        _ready = true;
        services.Runtime.ModelsChanged += _modelsChanged;
        services.Memory.Changed += _memoryChanged;
        // Opening the panel reads the figures now and keeps them fresh while it is open.
        _watch = services.Memory.Watch();
        ScheduleEstimate();
        WeightProbe = IsLanguageModel ? ProbeWeightFormatsAsync() : Task.CompletedTask;
    }

    /// <summary>How long the choices must stay unchanged before they are estimated.</summary>
    internal static TimeSpan EstimateDelay { get; set; } = TimeSpan.FromMilliseconds(250);

    /// <summary>The model to load.</summary>
    public LibraryEntryViewModel Entry { get; }

    /// <summary>The model generates text, so context, batching, KV cache and speculation apply (the other kinds load with defaults).</summary>
    public bool IsLanguageModel => !TaskKindClients.Handles(Entry.Entry.Kind) && Entry.Entry.Kind is not (ModelKindCatalog.Embeddings or ModelKindCatalog.Reranker);

    /// <summary>
    /// A speech synthesis model given as a language-model file or folder (OuteTTS with its WavTokenizer, Orpheus with SNAC) needs the codec decoder next to it; a package and a
    /// self-contained checkpoint folder (Kokoro) do not.
    /// </summary>
    public bool NeedsVocoder => Entry.Entry.Kind == Sharplight.Models.ModelKinds.SpeechSynthesis
        && !Entry.Entry.Path.EndsWith(".slm", StringComparison.OrdinalIgnoreCase) && !IsKokoroFolder(Entry.Entry.Path);

    private static bool IsKokoroFolder(string path)
    {
        string config = System.IO.Path.Combine(path, "config.json");
        return Directory.Exists(path) && File.Exists(config) && File.ReadAllText(config).Contains("\"istftnet\"", StringComparison.Ordinal);
    }

    /// <summary>The codec decoder file of a speech synthesis model (<see cref="NeedsVocoder"/>).</summary>
    [ObservableProperty]
    public partial string VocoderPath { get; set; } = "";

    /// <summary>
    /// An image model file that holds only the diffusion transformer (a FLUX or SD3 GGUF) needs its text encoders and VAE as companion files;
    /// the ones found beside it are filled in.
    /// </summary>
    public bool NeedsCompanions { get; }

    /// <summary>LoRA adapters can be merged into this model (an image generation model).</summary>
    public bool SupportsLoras => Entry.Entry.Kind == Sharplight.Models.ModelKinds.ImageGeneration;

    /// <summary>The LoRAs to merge into the model at load (<see cref="SupportsLoras"/>).</summary>
    public Image.LoraListViewModel Loras { get; }

    /// <summary>The companion files of a transformer-only image model (<see cref="NeedsCompanions"/>), in presentation order.</summary>
    public IReadOnlyList<CompanionFileViewModel> Companions { get; } = [];

    /// <summary>Devices.</summary>
    public IReadOnlyList<ComputeDeviceOption> Devices { get; }

    /// <summary>KV cache types.</summary>
    public IReadOnlyList<KvCacheChoice> KvCacheChoices => KvCacheChoice.All;

    /// <summary>Device to load on.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanOffload), nameof(CanUseBlockScaledFp8), nameof(BlockScaledFp8Hint), nameof(CanUseFp4Activations), nameof(Fp4ActivationsHint))]
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
    [NotifyPropertyChangedFor(nameof(CanTuneSteps), nameof(StepTuningHint))]
    public partial double Slots { get; set; }

    /// <summary>Offload what does not fit to the CPU.</summary>
    [ObservableProperty]
    public partial bool CpuOffload { get; set; }

    /// <summary>With CPU offload: leave routed experts that do not fit system memory on disk.</summary>
    [ObservableProperty]
    public partial bool ExpertsOnDisk { get; set; }

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

    /// <summary>Block-scaled FP8 tensor-core products for block-scaled FP8 weights (applies when <see cref="CanUseBlockScaledFp8"/>).</summary>
    [ObservableProperty]
    public partial bool BlockScaledFp8 { get; set; }

    /// <summary>FP4 activations for NVFP4 weights, the W4A4 GEMM (applies when <see cref="CanUseFp4Activations"/>).</summary>
    [ObservableProperty]
    public partial bool Fp4Activations { get; set; }

    /// <summary>When the floor of the adaptive step budget applies (applies when <see cref="CanTuneSteps"/>).</summary>
    [ObservableProperty]
    public partial StepFloorChoice StepFloor { get; set; }

    /// <summary>Rows per slot of the adaptive step budget; 0 keeps the engine's default (applies when <see cref="CanTuneSteps"/>).</summary>
    [ObservableProperty]
    public partial double StepRowsPerSequence { get; set; }

    /// <summary>The step floor choices.</summary>
    public IReadOnlyList<StepFloorChoice> StepFloorChoices => StepFloorChoice.All;

    /// <summary>The weight formats of the model, once read (null before, and for models that are not language models).</summary>
    public ModelWeightFormats? WeightFormats
    {
        get => _weightFormats;
        private set
        {
            if (SetProperty(ref _weightFormats, value))
            {
                OnPropertyChanged(nameof(CanUseBlockScaledFp8));
                OnPropertyChanged(nameof(BlockScaledFp8Hint));
                OnPropertyChanged(nameof(CanUseFp4Activations));
                OnPropertyChanged(nameof(Fp4ActivationsHint));
            }
        }
    }

    /// <summary>Reading the model's weight formats (for tests).</summary>
    internal Task WeightProbe { get; }

    /// <summary>The device has the block-scaled FP8 MMA (not turned off in the settings) and the model has block-scaled FP8 weights.</summary>
    public bool CanUseBlockScaledFp8 => DeviceHas(DeviceFeatures.BlockScaledFp8) && WeightFormats?.BlockScaledFp8 == true;

    /// <summary>The device has the block-scaled FP4 MMA (not turned off in the settings) and the model has NVFP4 weights.</summary>
    public bool CanUseFp4Activations => DeviceHas(DeviceFeatures.BlockScaledFp4) && WeightFormats?.Nvfp4 == true;

    /// <summary>What the FP8 toggle does, or why it does not apply.</summary>
    public string BlockScaledFp8Hint => Applicability(DeviceFeatures.BlockScaledFp8, WeightFormats?.BlockScaledFp8, "block-scaled FP8 weights",
        "Block-scaled FP8 weights are requantized to power-of-two scales and prompts run on the block-scaled FP8 tensor cores: a few percent faster, "
        + "perplexity within about +0.3 to +0.8 %. Off keeps the checkpoint's own scales.");

    /// <summary>What the FP4 toggle does, or why it does not apply.</summary>
    public string Fp4ActivationsHint => Applicability(DeviceFeatures.BlockScaledFp4, WeightFormats?.Nvfp4, "NVFP4 weights",
        "Prompts multiply NVFP4 weights with activations quantized to FP4 (W4A4) on the FP4 tensor cores: faster prefill, slightly less precise. "
        + "Off keeps 16-bit activations.");

    /// <summary>The step budget settings apply: several slots share the engine's steps.</summary>
    public bool CanTuneSteps => Slots > 1;

    /// <summary>What the step settings do, or why they do not apply.</summary>
    public string StepTuningHint => CanTuneSteps
        ? "How many prompt rows a step carries while few slots are busy: larger steps prefill a wave of prompts faster, smaller ones stream the first tokens sooner."
        : "Applies with two or more parallel slots (serving several requests at once).";

    private bool DeviceHas(string feature) => SelectedDevice is { } device && device.Features.Contains(feature)
        && !_services.Settings.Current.DisabledFeatures.Contains(feature, StringComparer.OrdinalIgnoreCase);

    private string Applicability(string feature, bool? modelHas, string weights, string description)
    {
        if (SelectedDevice is not { } device || !device.Features.Contains(feature))
        {
            return $"Not available on this device: it needs a GPU whose tensor cores apply block scales (CUDA, compute capability 12.x). {description}";
        }

        if (_services.Settings.Current.DisabledFeatures.Contains(feature, StringComparer.OrdinalIgnoreCase))
        {
            return $"Turned off for every model in Settings (device features). {description}";
        }

        return modelHas switch
        {
            null => $"Reading the model's weight formats… {description}",
            false => $"Does not apply: this model has no {weights}. {description}",
            true => description,
        };
    }

    private async Task ProbeWeightFormatsAsync()
    {
        try
        {
            var formats = await _services.Runtime.DescribeWeightFormatsAsync(ToRequest(), CancellationToken.None).ConfigureAwait(false);
            _services.Dispatcher.Post(() => WeightFormats = formats);
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            // The estimate reports an unreadable model; the toggles stay off.
            _services.Dispatcher.Post(() => WeightFormats = ModelWeightFormats.None);
        }
    }

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
            ExpertsOnDisk = ExpertsOnDisk && CpuOffload && CanOffload,
            NgramDraftTokens = Math.Max(0, (int)NgramDraftTokens),
            AllowReducedPrecision = AllowReducedPrecision,
            MtpDraftTokens = Math.Max(0, (int)MtpDraftTokens),
            BlockScaledFp8 = BlockScaledFp8 || !CanUseBlockScaledFp8,
            Fp4Activations = Fp4Activations || !CanUseFp4Activations,
            StepFloor = CanTuneSteps ? StepFloor.Value : null,
            StepRowsPerSequence = CanTuneSteps && StepRowsPerSequence >= 1 ? (int)StepRowsPerSequence : null,
        };
        var auxiliary = new List<(string Role, string Path)>();
        if (NeedsVocoder && VocoderPath.Trim().Length > 0)
        {
            auxiliary.Add((AuxiliaryRoles.Vocoder, VocoderPath.Trim()));
        }

        if (NeedsCompanions)
        {
            foreach (var companion in Companions.Where(companion => companion.Path.Trim().Length > 0))
            {
                auxiliary.Add((companion.Role, companion.Path.Trim()));
            }
        }

        if (SupportsLoras)
        {
            auxiliary.AddRange(LoraRoles.ToAuxiliary(Loras.Adapters));
        }

        return new ModelLoadRequest(ModelId.Trim(), Entry.Entry.Path, settings, _services.Settings.Current.DisabledFeatures,
            Capabilities: Entry.Entry.Capabilities, Kind: Entry.Entry.Kind, Auxiliary: auxiliary.Count > 0 ? auxiliary : null);
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

        if (NeedsVocoder && !File.Exists(VocoderPath.Trim()) && !Directory.Exists(VocoderPath.Trim()))
        {
            Status = "Choose the vocoder (the WavTokenizer .gguf or the SNAC folder that goes with this model).";
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
            // Only the latest status matters: the load reports per kernel and per tensor, the UI shows about 15 a second.
            using var log = new DispatchingProgress<LoadStatus>(_services.Dispatcher, ShowLoadStatus, static (_, _) => true);
            var request = ToRequest();
            var model = await Task.Run(() => _services.Runtime.LoadAsync(request, log, cancellation.Token), CancellationToken.None);
            log.Flush();
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

    private void ShowLoadStatus(LoadStatus status)
    {
        if (IsLoading)
        {
            Status = status.Text;
            Progress = status.Fraction;
        }
    }

    [RelayCommand]
    private async Task BrowseVocoderAsync()
    {
        var files = await _services.Platform.PickFilesAsync("Choose the vocoder", [new FileFilter("Codec files", ["*.gguf", "config.json", "pytorch_model.bin"])], allowMultiple: false);
        if (files.Count > 0)
        {
            // A checkpoint folder (SNAC) is named by any of its files.
            VocoderPath = files[0].EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) ? files[0] : System.IO.Path.GetDirectoryName(files[0]) ?? files[0];
        }
    }

    private async Task<string> PickCompanionAsync(string title, string current)
    {
        var files = await _services.Platform.PickFilesAsync(title, [new FileFilter("Model files", ["*.safetensors", "*.gguf"])], allowMultiple: false);
        return files.Count > 0 ? files[0] : current;
    }

    /// <summary>The first companion that is not an existing file, as the user knows it, or null.</summary>
    private string? MissingCompanion() => Companions.FirstOrDefault(companion => !File.Exists(companion.Path.Trim()))?.Label;

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
        _detached = true;
        _services.Runtime.ModelsChanged -= _modelsChanged;
        _services.Memory.Changed -= _memoryChanged;
        _watch.Dispose();
        _estimating?.Cancel();
    }

    /// <summary>
    /// Estimates the current choices once they stay unchanged for <see cref="EstimateDelay"/>, against <paramref name="memory"/> when given
    /// (new figures of the shared watch), else against figures read for it.
    /// </summary>
    private void ScheduleEstimate(IReadOnlyList<DeviceMemory>? memory = null)
    {
        lock (_scheduleLock)
        {
            // The new estimate is published before the old one is cancelled: whoever waits for Estimate never sees a cancelled one as the latest.
            var previous = _estimating;
            var cancellation = new CancellationTokenSource();
            _estimating = cancellation;
            Estimate = EstimateAsync(cancellation, memory);
            previous?.Cancel();
        }
    }

    private async Task EstimateAsync(CancellationTokenSource cancellation, IReadOnlyList<DeviceMemory>? known)
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
            var memory = known ?? await _advisor.GetMemoryAsync(token);
            Volatile.Write(ref _plannedMemory, memory);
            var footprint = await _services.Runtime.EstimateFootprintAsync(request, token);
            long experts = await _services.Runtime.EstimateExpertBytesAsync(request, token);
            var report = _advisor.Plan(footprint, request.Settings, memory);
            bool expertOffload = FitAdvisor.IsExpertOffload(report, experts);
            double? diskSpeed = report.ExpertsFromDisk
                ? FitAdvisor.DiskTokensPerSecond(report, await _services.Runtime.EstimateExpertActiveShareAsync(request, token))
                : null;
            token.ThrowIfCancellationRequested();
            FitError = "";
            // The verdict first; the longest contexts and the suggestions follow (a few more estimates of the same model).
            Fit = new FitEstimateViewModel(report, [], expertOffload: expertOffload, diskTokensPerSecond: diskSpeed)
            {
                ConfiguredContext = request.Settings.ContextLength,
            };
            var contexts = await _advisor.SizeContextAsync(request, memory, ContextSizer.Limit(Entry.Entry.TrainedContextLength), experts, token);
            var sizing = await _advisor.SizingAsync(request, report, contexts, memory, token);
            var suggestions = await _advisor.SuggestAsync(request, report, memory, Entry.Entry, _library(), token, contexts, experts);
            token.ThrowIfCancellationRequested();
            Fit = new FitEstimateViewModel(report, [.. suggestions.Select(suggestion => new FitSuggestionViewModel(suggestion, Apply))],
                [.. sizing.Select(row => new FitSuggestionViewModel(row, Apply))], contexts, expertOffload, diskSpeed)
            {
                ConfiguredContext = request.Settings.ContextLength,
            };
            if (Interlocked.Exchange(ref _pendingMemory, null) is { } newer && !DeviceMemoryWatch.Same(memory, newer))
            {
                ScheduleEstimate(newer);
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
        ExpertsOnDisk = settings.ExpertsOnDisk && CpuOffload;
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
