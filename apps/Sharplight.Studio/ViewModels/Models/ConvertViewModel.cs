using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>The convert panel: a local GGUF / checkpoint / ONNX model (with an optional vision projector) to a <c>.slm</c> package.</summary>
internal sealed partial class ConvertViewModel : ObservableObject
{
    private static readonly FileFilter s_models = new("Models", ["*.gguf", "*.onnx", "*.safetensors", "*.slm"]);
    private static readonly FileFilter s_projectors = new("Vision projectors", ["*.gguf"]);

    private readonly StudioServices _services;
    private readonly Action _converted;
    private readonly Action _closed;

    public ConvertViewModel(StudioServices services, Action converted, Action closed)
    {
        _services = services;
        _converted = converted;
        _closed = closed;
    }

    /// <summary>The model file or checkpoint directory.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ConvertCommand))]
    public partial string Source { get; set; } = "";

    /// <summary>Optional vision projector (<c>mmproj-*.gguf</c>).</summary>
    [ObservableProperty]
    public partial string Projector { get; set; } = "";

    /// <summary>The package to write.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ConvertCommand))]
    public partial string Destination { get; set; } = "";

    /// <summary>Converting.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(ConvertCommand))]
    public partial bool IsConverting { get; set; }

    /// <summary>Progress log.</summary>
    [ObservableProperty]
    public partial string Log { get; set; } = "";

    partial void OnSourceChanged(string value)
    {
        if (Destination.Length == 0 && value.Length > 0)
        {
            var directory = Path.GetDirectoryName(value.TrimEnd(Path.DirectorySeparatorChar)) ?? "";
            Destination = Path.Combine(directory, Path.GetFileNameWithoutExtension(value.TrimEnd(Path.DirectorySeparatorChar)) + ".slm");
        }
    }

    [RelayCommand]
    private async Task BrowseSourceAsync()
    {
        var files = await _services.Platform.PickFilesAsync("Model to convert", [s_models], allowMultiple: false);
        if (files.Count > 0)
        {
            Source = files[0];
        }
    }

    [RelayCommand]
    private async Task BrowseSourceFolderAsync()
    {
        if (await _services.Platform.PickFolderAsync("Checkpoint directory") is { } folder)
        {
            Source = folder;
        }
    }

    [RelayCommand]
    private async Task BrowseProjectorAsync()
    {
        var files = await _services.Platform.PickFilesAsync("Vision projector", [s_projectors], allowMultiple: false);
        if (files.Count > 0)
        {
            Projector = files[0];
        }
    }

    [RelayCommand]
    private async Task BrowseDestinationAsync()
    {
        var suggested = Destination.Length > 0 ? Path.GetFileName(Destination) : "model.slm";
        if (await _services.Platform.PickSaveFileAsync("Save package as", suggested, new FileFilter("Sharplight package", ["*.slm"])) is { } path)
        {
            Destination = path;
        }
    }

    private bool CanConvert() => !IsConverting && Source.Trim().Length > 0 && Destination.Trim().Length > 0;

    [RelayCommand(CanExecute = nameof(CanConvert))]
    private async Task ConvertAsync()
    {
        IsConverting = true;
        Log = "";
        var log = new Progress<string>(line => Log = Log.Length == 0 ? line : Log + "\n" + line);
        try
        {
            var projector = Projector.Trim().Length > 0 ? Projector.Trim() : null;
            await _services.Library.ConvertAsync(Source.Trim(), projector, Destination.Trim(), log, CancellationToken.None);
            Log += "\nDone: the package was added to the library.";
            _converted();
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            Log += "\nFailed: " + error.GetBaseException().Message;
        }
        finally
        {
            IsConverting = false;
        }
    }

    [RelayCommand]
    private void Close() => _closed();
}
