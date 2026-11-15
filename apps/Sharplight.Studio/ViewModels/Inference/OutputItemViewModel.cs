using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>An output in the gallery: the run's title, facts about it (metadata) and its files.</summary>
internal sealed partial class OutputItemViewModel : ObservableObject
{
    private readonly Action<OutputItemViewModel> _delete;
    private readonly IPlatformServices _platform;

    internal OutputItemViewModel(RunRecord record, PageArchive archive, IPlatformServices platform, Action<OutputItemViewModel> delete)
    {
        Record = record;
        _platform = platform;
        _delete = delete;
        Directory = archive.DirectoryOf(record);
        Files = [.. record.Files.Select(name => new OutputFileViewModel(archive.PathOf(record, name), platform))];
    }

    /// <summary>The run.</summary>
    public RunRecord Record { get; }

    /// <summary>The run's title.</summary>
    public string Title => Record.Title;

    /// <summary>The model that made it.</summary>
    public string Model => Record.Model;

    /// <summary>When it was made.</summary>
    public string When => Record.Created.LocalDateTime.ToString("g", System.Globalization.CultureInfo.CurrentCulture);

    /// <summary>A short excerpt of the output, or empty.</summary>
    public string Preview => Record.Preview ?? "";

    /// <summary>The directory of the output files.</summary>
    public string Directory { get; }

    /// <summary>The facts about the output, one <c>name: value</c> per entry.</summary>
    public IReadOnlyList<string> Facts => [.. Record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}")];

    /// <summary>The output files.</summary>
    public IReadOnlyList<OutputFileViewModel> Files { get; }

    /// <summary>Delete was clicked once; the next click deletes.</summary>
    [ObservableProperty]
    public partial bool ConfirmDelete { get; set; }

    /// <summary>Opens the primary file.</summary>
    [RelayCommand]
    private void Open()
    {
        if (Files.Count > 0)
        {
            Files[0].OpenCommand.Execute(null);
        }
        else
        {
            _platform.Open(Directory);
        }
    }

    /// <summary>Opens the folder with the files.</summary>
    [RelayCommand]
    private void OpenFolder() => _platform.Open(Directory);

    /// <summary>Exports the primary file.</summary>
    [RelayCommand]
    private async Task ExportAsync()
    {
        if (Files.Count > 0)
        {
            await Files[0].ExportCommand.ExecuteAsync(null);
        }
    }

    /// <summary>Deletes the output and its files after a second click.</summary>
    [RelayCommand]
    private void Delete()
    {
        if (!ConfirmDelete)
        {
            ConfirmDelete = true;
            return;
        }

        _delete(this);
    }
}
