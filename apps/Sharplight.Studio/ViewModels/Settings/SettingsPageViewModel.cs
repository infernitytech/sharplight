using System.Collections.ObjectModel;
using System.Reflection;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>The settings page: appearance, storage, defaults, engine options and data.</summary>
internal sealed partial class SettingsPageViewModel : PageViewModel
{
    private static readonly Dictionary<string, string> s_featureDescriptions = new(StringComparer.Ordinal)
    {
        [DeviceFeatures.MatrixF16] = "Accelerated f16 matrix multiply",
        [DeviceFeatures.MatrixBF16] = "Accelerated bf16 matrix multiply",
        [DeviceFeatures.MatrixInt8] = "Accelerated int8 matrix multiply",
        [DeviceFeatures.MatrixFp8] = "Accelerated fp8 (E4M3) matrix multiply",
        [DeviceFeatures.MatrixWorkgroup] = "Workgroup-wide matrix multiply",
        [DeviceFeatures.Float16] = "Native f16 arithmetic in kernels",
        [DeviceFeatures.BFloat16] = "Native bf16 types in kernels",
        [DeviceFeatures.Float8] = "Native 8-bit float types in kernels",
        [DeviceFeatures.IntDot] = "Packed int8 dot products",
        [DeviceFeatures.SubgroupOps] = "Subgroup reductions and shuffles",
        [DeviceFeatures.SubgroupSizeControl] = "Pinned subgroup size",
        [DeviceFeatures.MemoryBudget] = "Live memory budget including other processes",
    };

    private readonly StudioServices _services;
    private bool _initializing;

    public SettingsPageViewModel(StudioServices services)
    {
        _services = services;
        Load();
    }

    /// <inheritdoc/>
    public override string Title => "Settings";

    /// <inheritdoc/>
    public override string Icon => "IconSettings";

    /// <summary>Theme choices.</summary>
    public IReadOnlyList<ThemeChoice> Themes => ThemeChoice.All;

    /// <summary>Accent choices.</summary>
    public IReadOnlyList<AccentPalette> Accents => AccentPalette.All;

    /// <summary>KV cache choices.</summary>
    public IReadOnlyList<KvCacheChoice> KvCacheChoices => KvCacheChoice.All;

    /// <summary>Device features with their switches.</summary>
    public ObservableCollection<FeatureToggleViewModel> Features { get; } = [];

    /// <summary>The devices found.</summary>
    public IReadOnlyList<ComputeDeviceOption> Devices => _services.Runtime.Devices;

    /// <summary>Theme.</summary>
    [ObservableProperty]
    public partial ThemeChoice Theme { get; set; }

    /// <summary>Accent.</summary>
    [ObservableProperty]
    public partial AccentPalette Accent { get; set; }

    /// <summary>Model store chosen here (empty = default resolution).</summary>
    [ObservableProperty]
    public partial string ModelsDirectory { get; set; } = "";

    /// <summary>The store in effect.</summary>
    [ObservableProperty]
    public partial string StoreRoot { get; set; } = "";

    /// <summary>Where <see cref="StoreRoot"/> comes from.</summary>
    [ObservableProperty]
    public partial string StoreRootSource { get; set; } = "";

    /// <summary>Hugging Face token.</summary>
    [ObservableProperty]
    public partial string HuggingFaceToken { get; set; } = "";

    /// <summary>System prompt of new chats.</summary>
    [ObservableProperty]
    public partial string DefaultSystemPrompt { get; set; } = "";

    /// <summary>Thinking on by default.</summary>
    [ObservableProperty]
    public partial bool EnableThinking { get; set; }

    /// <summary>Default temperature.</summary>
    [ObservableProperty]
    public partial double Temperature { get; set; }

    /// <summary>Default top-p.</summary>
    [ObservableProperty]
    public partial double TopP { get; set; }

    /// <summary>Default top-k.</summary>
    [ObservableProperty]
    public partial double TopK { get; set; }

    /// <summary>Default min-p.</summary>
    [ObservableProperty]
    public partial double MinP { get; set; }

    /// <summary>Default answer limit.</summary>
    [ObservableProperty]
    public partial double MaxTokens { get; set; }

    /// <summary>Default device.</summary>
    [ObservableProperty]
    public partial ComputeDeviceOption? DefaultDevice { get; set; }

    /// <summary>Default context length.</summary>
    [ObservableProperty]
    public partial double ContextLength { get; set; }

    /// <summary>Default prompt chunk.</summary>
    [ObservableProperty]
    public partial double BatchTokens { get; set; }

    /// <summary>Default slots.</summary>
    [ObservableProperty]
    public partial double Slots { get; set; }

    /// <summary>Default KV cache type.</summary>
    [ObservableProperty]
    public partial KvCacheChoice KvCache { get; set; }

    /// <summary>Default CPU offload.</summary>
    [ObservableProperty]
    public partial bool CpuOffload { get; set; }

    /// <summary>Default n-gram draft length.</summary>
    [ObservableProperty]
    public partial double NgramDraftTokens { get; set; }

    /// <summary>Default MTP draft length.</summary>
    [ObservableProperty]
    public partial double MtpDraftTokens { get; set; }

    /// <summary>Device memory kept free, MiB.</summary>
    [ObservableProperty]
    public partial double ReserveMiB { get; set; }

    /// <summary>Resident model limit (0 = none).</summary>
    [ObservableProperty]
    public partial double MaxLoadedModels { get; set; }

    /// <summary>Idle unload timeout in minutes (0 = never).</summary>
    [ObservableProperty]
    public partial double IdleUnloadMinutes { get; set; }

    /// <summary>Pin models loaded by hand.</summary>
    [ObservableProperty]
    public partial bool PinManualLoads { get; set; }

    /// <summary>Load library models on demand.</summary>
    [ObservableProperty]
    public partial bool LoadOnDemand { get; set; }

    /// <summary>Default for the reduced-precision opt-in.</summary>
    [ObservableProperty]
    public partial bool AllowReducedPrecision { get; set; }

    /// <summary>Compile kernels after a download.</summary>
    [ObservableProperty]
    public partial bool PrecompileAfterDownload { get; set; }

    /// <summary>Keep the source files after converting.</summary>
    [ObservableProperty]
    public partial bool KeepSourceFiles { get; set; }

    /// <summary>Feedback of the last action.</summary>
    [ObservableProperty]
    public partial string? Message { get; set; }

    /// <summary>App version.</summary>
    public string Version => typeof(SettingsPageViewModel).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion.Split('+')[0] ?? "dev";

    /// <summary>.NET runtime.</summary>
    public string Runtime => System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription;

    /// <summary>Features disabled through <c>SHARPLIGHT_DISABLE_FEATURES</c>.</summary>
    public string EnvironmentDisabled => DeviceFeatures.DisabledByEnvironment.Count == 0 ? "" : "Disabled by SHARPLIGHT_DISABLE_FEATURES: " + string.Join(", ", DeviceFeatures.DisabledByEnvironment);

    partial void OnThemeChanged(ThemeChoice value)
    {
        Save();
        if (!_initializing)
        {
            _services.Theme.Apply(value.Value, Accent.Name);
        }
    }

    partial void OnAccentChanged(AccentPalette value)
    {
        Save();
        if (!_initializing)
        {
            _services.Theme.Apply(Theme.Value, value.Name);
        }
    }

    partial void OnModelsDirectoryChanged(string value) => Save();

    partial void OnHuggingFaceTokenChanged(string value) => Save();

    partial void OnDefaultSystemPromptChanged(string value) => Save();

    partial void OnEnableThinkingChanged(bool value) => Save();

    partial void OnTemperatureChanged(double value) => Save();

    partial void OnTopPChanged(double value) => Save();

    partial void OnTopKChanged(double value) => Save();

    partial void OnMinPChanged(double value) => Save();

    partial void OnMaxTokensChanged(double value) => Save();

    partial void OnDefaultDeviceChanged(ComputeDeviceOption? value) => Save();

    partial void OnContextLengthChanged(double value) => Save();

    partial void OnBatchTokensChanged(double value) => Save();

    partial void OnSlotsChanged(double value) => Save();

    partial void OnKvCacheChanged(KvCacheChoice value) => Save();

    partial void OnCpuOffloadChanged(bool value) => Save();

    partial void OnNgramDraftTokensChanged(double value) => Save();

    partial void OnAllowReducedPrecisionChanged(bool value) => Save();

    partial void OnMtpDraftTokensChanged(double value) => Save();

    partial void OnReserveMiBChanged(double value) => Save();

    partial void OnMaxLoadedModelsChanged(double value) => Save();

    partial void OnIdleUnloadMinutesChanged(double value) => Save();

    partial void OnPinManualLoadsChanged(bool value) => Save();

    partial void OnLoadOnDemandChanged(bool value) => Save();

    partial void OnPrecompileAfterDownloadChanged(bool value) => Save();

    partial void OnKeepSourceFilesChanged(bool value) => Save();

    [RelayCommand]
    private async Task BrowseModelsDirectoryAsync()
    {
        var folder = await _services.Platform.PickFolderAsync("Model store");
        if (folder is not null)
        {
            ModelsDirectory = folder;
        }
    }

    [RelayCommand]
    private void ResetModelsDirectory() => ModelsDirectory = "";

    [RelayCommand]
    private void OpenModelsDirectory() => _services.Platform.Open(StoreRoot);

    [RelayCommand]
    private void ResetDefaults()
    {
        var defaults = new AppSettings();
        _services.Settings.Update(current => current with
        {
            DefaultLoad = defaults.DefaultLoad,
            DefaultSampling = defaults.DefaultSampling,
            DefaultSystemPrompt = defaults.DefaultSystemPrompt,
            EnableThinking = defaults.EnableThinking,
            DisabledFeatures = [],
        });
        Load();
        Message = "Defaults restored.";
    }

    [RelayCommand]
    private async Task ClearStatisticsAsync()
    {
        await _services.Stats.ClearAsync();
        Message = "Statistics cleared.";
    }

    private void Load()
    {
        _initializing = true;
        var settings = _services.Settings.Current;
        Theme = ThemeChoice.All.First(choice => choice.Value == settings.Theme);
        Accent = AccentPalette.Get(settings.Accent);
        ModelsDirectory = settings.ModelsDirectory ?? "";
        HuggingFaceToken = settings.HuggingFaceToken ?? "";
        DefaultSystemPrompt = settings.DefaultSystemPrompt;
        EnableThinking = settings.EnableThinking;
        var sampling = settings.DefaultSampling;
        Temperature = sampling.Temperature;
        TopP = sampling.TopP;
        TopK = sampling.TopK;
        MinP = sampling.MinP;
        MaxTokens = sampling.MaxTokens;
        var load = settings.DefaultLoad;
        DefaultDevice = Devices.FirstOrDefault(device => device.Id == load.DeviceId) ?? Devices.FirstOrDefault();
        ContextLength = load.ContextLength;
        BatchTokens = load.BatchTokens;
        Slots = load.Slots;
        KvCache = KvCacheChoice.All.FirstOrDefault(choice => choice.Value == load.KvCacheType) ?? KvCacheChoice.All[0];
        CpuOffload = load.CpuOffload;
        NgramDraftTokens = load.NgramDraftTokens;
        AllowReducedPrecision = load.AllowReducedPrecision;
        MtpDraftTokens = load.MtpDraftTokens;
        var residency = settings.Residency;
        ReserveMiB = residency.ReserveMiB;
        MaxLoadedModels = residency.MaxLoadedModels;
        IdleUnloadMinutes = residency.IdleUnloadMinutes;
        PinManualLoads = residency.PinManualLoads;
        LoadOnDemand = residency.LoadOnDemand;
        PrecompileAfterDownload = settings.PrecompileAfterDownload;
        KeepSourceFiles = settings.KeepSourceFiles;
        Features.Clear();
        var disabled = settings.DisabledFeatures.ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (var feature in DeviceFeatures.All)
        {
            Features.Add(new FeatureToggleViewModel(feature, s_featureDescriptions.GetValueOrDefault(feature, feature), !disabled.Contains(feature), Save));
        }

        RefreshStore();
        _initializing = false;
    }

    private void RefreshStore()
    {
        StoreRoot = _services.Settings.StoreRoot;
        StoreRootSource = _services.Settings.StoreRootSource;
    }

    private void Save()
    {
        if (_initializing)
        {
            return;
        }

        _services.Settings.Update(current => current with
        {
            Theme = Theme.Value,
            Accent = Accent.Name,
            ModelsDirectory = string.IsNullOrWhiteSpace(ModelsDirectory) ? null : ModelsDirectory.Trim(),
            HuggingFaceToken = string.IsNullOrWhiteSpace(HuggingFaceToken) ? null : HuggingFaceToken.Trim(),
            DefaultSystemPrompt = DefaultSystemPrompt,
            EnableThinking = EnableThinking,
            DefaultSampling = current.DefaultSampling with
            {
                Temperature = Math.Clamp(Temperature, 0, 2),
                TopP = Math.Clamp(TopP, 0, 1),
                TopK = Math.Max(0, (int)TopK),
                MinP = Math.Clamp(MinP, 0, 1),
                MaxTokens = Math.Max(0, (int)MaxTokens),
            },
            DefaultLoad = current.DefaultLoad with
            {
                DeviceId = DefaultDevice?.Id,
                ContextLength = Math.Max(256, (int)ContextLength),
                BatchTokens = Math.Max(16, (int)BatchTokens),
                Slots = Math.Max(1, (int)Slots),
                KvCacheType = KvCache.Value,
                CpuOffload = CpuOffload,
                NgramDraftTokens = Math.Max(0, (int)NgramDraftTokens),
                AllowReducedPrecision = AllowReducedPrecision,
                MtpDraftTokens = Math.Max(0, (int)MtpDraftTokens),
            },
            Residency = current.Residency with
            {
                ReserveMiB = Math.Max(0, (int)ReserveMiB),
                MaxLoadedModels = Math.Max(0, (int)MaxLoadedModels),
                IdleUnloadMinutes = Math.Max(0, (int)IdleUnloadMinutes),
                PinManualLoads = PinManualLoads,
                LoadOnDemand = LoadOnDemand,
            },
            DisabledFeatures = [.. Features.Where(feature => !feature.IsEnabled).Select(feature => feature.Name)],
            PrecompileAfterDownload = PrecompileAfterDownload,
            KeepSourceFiles = KeepSourceFiles,
        });
        RefreshStore();
    }
}
