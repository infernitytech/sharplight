using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>One companion file of a transformer-only image model in the load panel: its role, what it is, and the chosen file.</summary>
internal sealed partial class CompanionFileViewModel : ObservableObject
{
    private readonly Func<string, string, Task<string>> _pick;

    /// <param name="role">The auxiliary role the file is loaded in.</param>
    /// <param name="label">What it is, for people.</param>
    /// <param name="example">Typical file names.</param>
    /// <param name="path">The file found beside the model, or "".</param>
    /// <param name="pick">Shows a file picker (title, current path) and returns the chosen path or the current one.</param>
    public CompanionFileViewModel(string role, string label, string example, string path, Func<string, string, Task<string>> pick)
    {
        Role = role;
        Label = label;
        Example = example;
        Path = path;
        _pick = pick;
    }

    /// <summary>The auxiliary role.</summary>
    public string Role { get; }

    /// <summary>What the file is (e.g. "Text encoder (CLIP-L)").</summary>
    public string Label { get; }

    /// <summary>Typical file names, shown as the placeholder.</summary>
    public string Example { get; }

    /// <summary>The chosen file.</summary>
    [ObservableProperty]
    public partial string Path { get; set; }

    [RelayCommand]
    private async Task BrowseAsync() => Path = await _pick($"Choose the {Label}", Path);
}
