using CommunityToolkit.Mvvm.Input;

namespace Sharplight.Studio.ViewModels.Audio;

/// <summary>One stem of a separation result with its own playback and export.</summary>
internal sealed partial class StemViewModel
{
    private readonly AudioPageViewModel _page;

    /// <param name="page">The page whose player and exporter the stem uses.</param>
    /// <param name="name">The stem's name.</param>
    /// <param name="path">The stored WAV file.</param>
    public StemViewModel(AudioPageViewModel page, string name, string path)
    {
        _page = page;
        Name = name;
        Path = path;
    }

    /// <summary>The stem's name.</summary>
    public string Name { get; }

    /// <summary>The stored WAV file.</summary>
    public string Path { get; }

    /// <summary>Plays the stem, or stops it when it is playing.</summary>
    [RelayCommand]
    private Task PlayAsync() => _page.PlayFileAsync(Path);

    /// <summary>Saves the stem where the user chooses.</summary>
    [RelayCommand]
    private Task SaveAsync() => _page.SaveFileAsync(Path, $"{Name}.wav");
}
