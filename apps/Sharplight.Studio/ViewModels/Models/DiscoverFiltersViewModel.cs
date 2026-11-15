using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Hub;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>
/// The filters and sort order of Discover: model type, format, quantization, size, "fits this machine", license, gated and the sort
/// order. Every change is saved in the settings (<see cref="AppSettings.Discover"/>) and raises <see cref="FiltersChanged"/>.
/// </summary>
internal sealed partial class DiscoverFiltersViewModel : ObservableObject
{
    private static readonly string[] s_licenses =
        ["apache-2.0", "mit", "bsd-3-clause", "cc-by-4.0", "cc-by-nc-4.0", "cc-by-sa-4.0", "openrail", "llama3.1", "llama3.2", "gemma", "other"];

    private readonly ISettingsService _settings;
    private bool _loading;

    /// <param name="settings">The settings the choices are read from and saved to.</param>
    public DiscoverFiltersViewModel(ISettingsService settings)
    {
        _settings = settings;
        Tasks = [new("All model types", null), .. ModelTasks.All.Select(task => new FilterOption<ModelTask?>(ModelTaskTitles.Title(task), task))];
        Formats =
        [
            new("Any format", null),
            new("GGUF", ModelFormat.Gguf),
            new("Safetensors", ModelFormat.Safetensors),
            new("Diffusers", ModelFormat.Diffusers),
            new("ONNX", ModelFormat.Onnx),
            new("whisper.cpp", ModelFormat.WhisperCpp),
            new(".slm package", ModelFormat.Slm),
        ];
        Quantizations = [new("Any quantization", null), .. DiscoverFilter.Quantizations.Select(name => new FilterOption<string?>(name, name))];
        Sizes = [.. Enum.GetValues<ModelSizeFilter>().Select(size => new FilterOption<ModelSizeFilter>(DiscoverFilter.Describe(size), size))];
        Licenses = [new("Any license", null), .. s_licenses.Select(name => new FilterOption<string?>(name, name))];
        GatedChoices = [new("Gated and open", GatedFilter.Any), new("Hide gated", GatedFilter.Hide), new("Only gated", GatedFilter.Only)];
        Sorts =
        [
            new("Most downloaded", HubSortOrder.Downloads),
            new("Trending", HubSortOrder.Trending),
            new("Most liked", HubSortOrder.Likes),
            new("Recently updated", HubSortOrder.Modified),
            new("Newest", HubSortOrder.Newest),
        ];
        Load(settings.Current.Discover);
    }

    /// <summary>Raised after a choice changed (and was saved); the search or highlights are asked again.</summary>
    public event EventHandler? FiltersChanged;

    /// <summary>Model type choices (every <see cref="ModelTask"/>).</summary>
    public IReadOnlyList<FilterOption<ModelTask?>> Tasks { get; }

    /// <summary>Format choices.</summary>
    public IReadOnlyList<FilterOption<ModelFormat?>> Formats { get; }

    /// <summary>Quantization choices.</summary>
    public IReadOnlyList<FilterOption<string?>> Quantizations { get; }

    /// <summary>Parameter range choices.</summary>
    public IReadOnlyList<FilterOption<ModelSizeFilter>> Sizes { get; }

    /// <summary>License choices.</summary>
    public IReadOnlyList<FilterOption<string?>> Licenses { get; }

    /// <summary>Gated choices.</summary>
    public IReadOnlyList<FilterOption<GatedFilter>> GatedChoices { get; }

    /// <summary>Sort order choices.</summary>
    public IReadOnlyList<FilterOption<HubSortOrder>> Sorts { get; }

    /// <summary>The chosen model type.</summary>
    [ObservableProperty]
    public partial FilterOption<ModelTask?> SelectedTask { get; set; } = null!;

    /// <summary>The chosen format.</summary>
    [ObservableProperty]
    public partial FilterOption<ModelFormat?> SelectedFormat { get; set; } = null!;

    /// <summary>The chosen quantization family.</summary>
    [ObservableProperty]
    public partial FilterOption<string?> SelectedQuantization { get; set; } = null!;

    /// <summary>The chosen parameter range.</summary>
    [ObservableProperty]
    public partial FilterOption<ModelSizeFilter> SelectedSize { get; set; } = null!;

    /// <summary>Only models whose smallest download likely fits this machine.</summary>
    [ObservableProperty]
    public partial bool FitsThisMachine { get; set; }

    /// <summary>The chosen license.</summary>
    [ObservableProperty]
    public partial FilterOption<string?> SelectedLicense { get; set; } = null!;

    /// <summary>The chosen gated mode.</summary>
    [ObservableProperty]
    public partial FilterOption<GatedFilter> SelectedGated { get; set; } = null!;

    /// <summary>The chosen sort order.</summary>
    [ObservableProperty]
    public partial FilterOption<HubSortOrder> SelectedSort { get; set; } = null!;

    /// <summary>The choices as settings.</summary>
    public DiscoverFilterSettings Current => new()
    {
        Task = SelectedTask.Value,
        Format = SelectedFormat.Value,
        Quantization = SelectedQuantization.Value,
        Size = SelectedSize.Value,
        FitsThisMachine = FitsThisMachine,
        License = SelectedLicense.Value,
        Gated = SelectedGated.Value,
        Sort = SelectedSort.Value,
    };

    /// <summary>A choice other than the default is set (the sort order included).</summary>
    public bool HasActiveFilters => Current != new DiscoverFilterSettings();

    /// <summary>Sets every choice back to its default.</summary>
    [RelayCommand]
    private void Reset() => Apply(new DiscoverFilterSettings());

    /// <summary>Selects the choices of <paramref name="settings"/> (saved and announced like a change by the user).</summary>
    /// <param name="settings">The filters.</param>
    public void Apply(DiscoverFilterSettings settings)
    {
        Load(settings);
        Save();
    }

    private void Load(DiscoverFilterSettings settings)
    {
        _loading = true;
        try
        {
            SelectedTask = Tasks.First(option => option.Value == settings.Task);
            SelectedFormat = Formats.First(option => option.Value == settings.Format);
            SelectedQuantization = Quantizations.FirstOrDefault(option => option.Value == settings.Quantization) ?? Quantizations[0];
            SelectedSize = Sizes.First(option => option.Value == settings.Size);
            FitsThisMachine = settings.FitsThisMachine;
            SelectedLicense = Licenses.FirstOrDefault(option => option.Value == settings.License) ?? Licenses[0];
            SelectedGated = GatedChoices.First(option => option.Value == settings.Gated);
            SelectedSort = Sorts.First(option => option.Value == settings.Sort);
        }
        finally
        {
            _loading = false;
        }
    }

    private void Save()
    {
        var current = Current;
        if (_settings.Current.Discover != current)
        {
            _settings.Update(settings => settings with { Discover = current });
        }

        OnPropertyChanged(nameof(HasActiveFilters));
        FiltersChanged?.Invoke(this, EventArgs.Empty);
    }

    partial void OnSelectedTaskChanged(FilterOption<ModelTask?> value) => Changed();

    partial void OnSelectedFormatChanged(FilterOption<ModelFormat?> value) => Changed();

    partial void OnSelectedQuantizationChanged(FilterOption<string?> value) => Changed();

    partial void OnSelectedSizeChanged(FilterOption<ModelSizeFilter> value) => Changed();

    partial void OnFitsThisMachineChanged(bool value) => Changed();

    partial void OnSelectedLicenseChanged(FilterOption<string?> value) => Changed();

    partial void OnSelectedGatedChanged(FilterOption<GatedFilter> value) => Changed();

    partial void OnSelectedSortChanged(FilterOption<HubSortOrder> value) => Changed();

    private void Changed()
    {
        if (!_loading)
        {
            Save();
        }
    }
}
