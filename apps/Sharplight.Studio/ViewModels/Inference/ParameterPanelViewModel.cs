using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>
/// The parameter panel every inference page shares: the parameters of the page's task options, and presets of their values saved as
/// JSON (<c>presets.json</c> in the page's directory; built-in presets ship with the page). The values of the last run are restored
/// when the page opens. <see cref="ParameterPanelViewModel{TOptions}"/> adds the mapping to the options type.
/// </summary>
internal partial class ParameterPanelViewModel : ObservableObject
{
    private readonly string _file;
    private readonly ParameterPresetFile _stored;
    private readonly IReadOnlyList<ParameterPreset> _builtIn;
    private bool _applying;

    /// <param name="storage">The page storage.</param>
    /// <param name="pageId">The page's id.</param>
    /// <param name="parameters">The parameters, in display order.</param>
    /// <param name="builtIn">Presets that ship with the page.</param>
    public ParameterPanelViewModel(IPageStorage storage, string pageId, IReadOnlyList<ParameterViewModel> parameters, IReadOnlyList<ParameterPreset>? builtIn = null)
    {
        Parameters = parameters;
        _builtIn = builtIn ?? [];
        _file = Path.Combine(storage.DirectoryFor(pageId), "presets.json");
        _stored = PageJson.Read(_file, StudioJsonContext.Default.ParameterPresetFile);
        RebuildPresets();
        Apply(_stored.Last);
        foreach (var parameter in parameters)
        {
            parameter.PropertyChanged += (_, _) =>
            {
                if (!_applying)
                {
                    IsModified = true;
                }
            };
        }

        IsModified = false;
    }

    /// <summary>The parameters, in display order.</summary>
    public IReadOnlyList<ParameterViewModel> Parameters { get; }

    /// <summary>The built-in presets, then the user's.</summary>
    public ObservableCollection<ParameterPreset> Presets { get; } = [];

    /// <summary>The preset last applied or saved.</summary>
    [ObservableProperty]
    public partial ParameterPreset? SelectedPreset { get; set; }

    /// <summary>The name a new preset is saved under.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SavePresetCommand))]
    public partial string PresetName { get; set; } = "";

    /// <summary>The values differ from the selected preset (or from the defaults).</summary>
    [ObservableProperty]
    public partial bool IsModified { get; set; }

    /// <summary>The values of every parameter by key, as presets store them.</summary>
    public Dictionary<string, string> Snapshot() => Parameters.ToDictionary(parameter => parameter.Key, parameter => parameter.Serialize());

    /// <summary>Sets the parameters named in <paramref name="values"/>; the others keep their value.</summary>
    /// <param name="values">Values by parameter key.</param>
    public void Apply(IReadOnlyDictionary<string, string> values)
    {
        _applying = true;
        try
        {
            foreach (var parameter in Parameters)
            {
                if (values.TryGetValue(parameter.Key, out var text))
                {
                    parameter.Deserialize(text);
                }
            }
        }
        finally
        {
            _applying = false;
        }

        IsModified = false;
    }

    /// <summary>Remembers the current values for the next time the page opens (called when a run starts).</summary>
    public void Persist()
    {
        _stored.Last = Snapshot();
        Save();
    }

    /// <summary>The value of the bool parameter <paramref name="key"/>.</summary>
    public bool GetBool(string key) => Find<BoolParameterViewModel>(key).Value;

    /// <summary>The value of the number parameter <paramref name="key"/>.</summary>
    public double GetNumber(string key) => Find<NumberParameterViewModel>(key).Value;

    /// <summary>The value of the number parameter <paramref name="key"/>, rounded.</summary>
    public int GetInt(string key) => Find<NumberParameterViewModel>(key).IntValue;

    /// <summary>The chosen value of the choice parameter <paramref name="key"/>.</summary>
    public string GetChoice(string key) => Find<ChoiceParameterViewModel>(key).Value;

    /// <summary>The text of the text parameter <paramref name="key"/>.</summary>
    public string GetText(string key) => Find<TextParameterViewModel>(key).Value;

    /// <summary>Shows or hides a parameter (pages hide what the chosen model does not use).</summary>
    /// <param name="key">The parameter.</param>
    /// <param name="visible">Whether it is shown.</param>
    public void SetVisible(string key, bool visible) => Find<ParameterViewModel>(key).IsVisibleNow = visible;

    partial void OnSelectedPresetChanged(ParameterPreset? value)
    {
        if (value is not null)
        {
            PresetName = value.BuiltIn ? "" : value.Name;
            Apply(value.Values);
        }
    }

    private bool CanSave() => PresetName.Trim().Length > 0;

    /// <summary>Saves the current values as a preset named <see cref="PresetName"/> (replacing one of that name).</summary>
    [RelayCommand(CanExecute = nameof(CanSave))]
    private void SavePreset()
    {
        string name = PresetName.Trim();
        _stored.Presets.RemoveAll(preset => preset.Name == name);
        var saved = new ParameterPreset(name, Snapshot());
        _stored.Presets.Add(saved);
        Save();
        RebuildPresets();
        _applying = true;
        SelectedPreset = Presets.First(preset => preset.Name == name && !preset.BuiltIn);
        _applying = false;
        IsModified = false;
    }

    /// <summary>Deletes the selected preset (built-in ones stay).</summary>
    [RelayCommand]
    private void DeletePreset()
    {
        if (SelectedPreset is { BuiltIn: false } preset)
        {
            _stored.Presets.Remove(preset);
            Save();
            RebuildPresets();
        }
    }

    /// <summary>Sets every parameter to its default.</summary>
    [RelayCommand]
    private void ResetToDefaults()
    {
        _applying = true;
        foreach (var parameter in Parameters)
        {
            parameter.Reset();
        }

        _applying = false;
        SelectedPreset = null;
        IsModified = false;
    }

    private void RebuildPresets()
    {
        Presets.Clear();
        foreach (var preset in _builtIn.Concat(_stored.Presets))
        {
            Presets.Add(preset);
        }

        SelectedPreset = null;
    }

    private void Save()
    {
        PageJson.Write(_file, _stored, StudioJsonContext.Default.ParameterPresetFile);
    }

    private T Find<T>(string key)
        where T : ParameterViewModel
        => Parameters.OfType<T>().FirstOrDefault(parameter => parameter.Key == key)
           ?? throw new KeyNotFoundException($"The page has no {typeof(T).Name.Replace("ViewModel", "")} '{key}'.");
}
