using System.Globalization;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Services.Gallery;

namespace Sharplight.Studio.ViewModels.Gallery;

/// <summary>One image of the gallery: its stored metadata and a thumbnail that loads when the list first asks for it.</summary>
internal sealed partial class GalleryItemViewModel : ObservableObject
{
    private readonly ThumbnailCache _thumbnails;
    private readonly string _path;

    /// <param name="entry">The stored entry.</param>
    /// <param name="path">The PNG's absolute path.</param>
    /// <param name="thumbnails">Where thumbnails are decoded and kept.</param>
    public GalleryItemViewModel(GalleryEntry entry, string path, ThumbnailCache thumbnails)
    {
        Entry = entry;
        _path = path;
        _thumbnails = thumbnails;
    }

    /// <summary>The stored entry.</summary>
    public GalleryEntry Entry { get; }

    /// <summary>The first words of the prompt, for the tooltip and the list.</summary>
    public string Title => Entry.Prompt.Length > 60 ? Entry.Prompt[..60] + "…" : Entry.Prompt;

    /// <summary>Model, size and seed, for the tooltip.</summary>
    public string Caption => $"{Entry.Model} · {Entry.Width} × {Entry.Height} · seed {Entry.Seed.ToString(CultureInfo.InvariantCulture)}\n{Entry.Created.LocalDateTime:g}\n{Title}";

    /// <summary>The thumbnail; null until it is decoded (decoding starts on the first read, on a worker thread).</summary>
    public Bitmap? Thumbnail => _thumbnails.Get(this, _path);

    /// <summary>The user asked to delete this image and has not confirmed yet.</summary>
    [ObservableProperty]
    public partial bool IsConfirmingDelete { get; set; }

    internal void ThumbnailReady() => OnPropertyChanged(nameof(Thumbnail));
}
