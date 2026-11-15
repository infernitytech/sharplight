using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>One output file of a run in the gallery: it can be opened in the system's program or exported.</summary>
internal sealed partial class OutputFileViewModel
{
    private readonly IPlatformServices _platform;

    internal OutputFileViewModel(string path, IPlatformServices platform)
    {
        Path = path;
        _platform = platform;
    }

    /// <summary>The file.</summary>
    public string Path { get; }

    /// <summary>The file's name.</summary>
    public string Name => System.IO.Path.GetFileName(Path);

    /// <summary>The file's size.</summary>
    public string Size => File.Exists(Path) ? ByteFormat.Format(new FileInfo(Path).Length) : "missing";

    /// <summary>Opens the file in the program the system associates with it.</summary>
    [RelayCommand]
    private void Open() => _platform.Open(Path);

    /// <summary>Saves a copy where the user chooses.</summary>
    [RelayCommand]
    private async Task ExportAsync()
    {
        if (File.Exists(Path))
        {
            await Exporter.CopyAsync(_platform, Path);
        }
    }
}
