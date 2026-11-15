using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.Services.Platform;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>
/// The image input every image-taking page shares (vision pages, the image page's input image and mask): choose a file or drop one
/// on the block, see a preview with its size, clear it. The page reads <see cref="Image"/>.
/// </summary>
internal sealed partial class ImageInputViewModel : ObservableObject
{
    private readonly IPlatformServices _platform;
    private readonly Action<string> _notice;
    private readonly Action? _changed;

    /// <param name="platform">The file picker.</param>
    /// <param name="notice">Receives what to tell the user when a file cannot be used (and an empty text when one was taken).</param>
    /// <param name="title">The block's title.</param>
    /// <param name="hint">What to do, shown while there is no image.</param>
    /// <param name="changed">Called after the image changed.</param>
    public ImageInputViewModel(IPlatformServices platform, Action<string> notice, string title, string hint, Action? changed = null)
    {
        _platform = platform;
        _notice = notice;
        Title = title;
        Hint = hint;
        _changed = changed;
    }

    /// <summary>The block's title.</summary>
    public string Title { get; }

    /// <summary>What to do, shown while there is no image.</summary>
    public string Hint { get; }

    /// <summary>The image, or null.</summary>
    public ImageInput? Image { get; private set; }

    /// <summary>An image is chosen.</summary>
    public bool HasImage => Image is not null;

    /// <summary>The image as a PNG, for the preview.</summary>
    [ObservableProperty]
    public partial byte[]? Preview { get; private set; }

    /// <summary>The path of the file the image came from, or empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Name))]
    public partial string Path { get; private set; } = "";

    /// <summary>The name shown for the image.</summary>
    public string Name => Path.Length == 0 ? "" : System.IO.Path.GetFileName(Path);

    /// <summary>Width and height of the image.</summary>
    [ObservableProperty]
    public partial string Summary { get; private set; } = "";

    /// <summary>Reads <paramref name="path"/> as the image.</summary>
    /// <param name="path">An image file.</param>
    /// <returns>Whether the file was usable (otherwise the notice says why).</returns>
    public async Task<bool> SetFileAsync(string path)
    {
        if (!File.Exists(path))
        {
            _notice($"{path} does not exist.");
            return false;
        }

        if (!ImageFiles.IsImage(path))
        {
            _notice($"{System.IO.Path.GetFileName(path)} is not an image file (PNG, JPEG, WebP, BMP or GIF).");
            return false;
        }

        try
        {
            var image = await Task.Run(() => ImageFiles.Read(path));
            Set(image, path);
            return true;
        }
        catch (Exception error) when (error is IOException or InvalidDataException or NotSupportedException)
        {
            _notice($"{System.IO.Path.GetFileName(path)} cannot be read: {error.Message}");
            return false;
        }
    }

    /// <summary>Takes a decoded image (for example a generated one) as the input.</summary>
    /// <param name="image">The image.</param>
    /// <param name="path">The file it came from, or a name to show.</param>
    public void Set(ImageInput image, string path)
    {
        _notice("");
        Image = image;
        Path = path;
        Summary = $"{image.Width} × {image.Height} pixels";
        Preview = ImageFiles.ToPng(image);
        OnPropertyChanged(nameof(HasImage));
        _changed?.Invoke();
    }

    /// <summary>Removes the image.</summary>
    [RelayCommand]
    public void Clear()
    {
        if (Image is null)
        {
            return;
        }

        Image = null;
        Preview = null;
        Path = "";
        Summary = "";
        OnPropertyChanged(nameof(HasImage));
        _changed?.Invoke();
    }

    /// <summary>Opens the file picker.</summary>
    [RelayCommand]
    private async Task ChooseAsync()
    {
        var files = await _platform.PickFilesAsync($"Choose {Title.ToLowerInvariant()}", [ImageFiles.Filter, new FileFilter("All files", ["*"])], allowMultiple: false);
        if (files.Count > 0)
        {
            await SetFileAsync(files[0]);
        }
    }
}
