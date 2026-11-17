using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services.Gallery;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.ViewModels.Inference;

namespace Sharplight.Studio.ViewModels.Gallery;

/// <summary>
/// The gallery of every generated image: newest first, filtered by model and prompt text, with open, reuse settings, copy prompt,
/// open folder, export and delete (confirmed). Entries are read from the store off the UI thread; the list is virtualized by the view
/// and the thumbnails load lazily (<see cref="ThumbnailCache"/>), so thousands of images stay cheap.
/// </summary>
internal sealed partial class ImageGalleryViewModel : ObservableObject
{
    /// <summary>The filter value that shows every model.</summary>
    public const string AllModels = "All models";

    private readonly ImageGalleryStore _store;
    private readonly IPlatformServices _platform;
    private readonly IUiDispatcher _dispatcher;
    private readonly ThumbnailCache _thumbnails;
    private readonly Action<GalleryEntry> _reuse;
    private readonly Action<GalleryItemViewModel> _open;
    private readonly List<GalleryItemViewModel> _all = [];
    private readonly string _noun;

    /// <param name="store">The images on disk.</param>
    /// <param name="platform">Clipboard, file manager and save dialog.</param>
    /// <param name="dispatcher">Runs the results of worker threads on the UI thread.</param>
    /// <param name="open">Shows an image large.</param>
    /// <param name="reuse">Fills the page's form with an entry's settings.</param>
    /// <param name="noun">What the gallery holds, in the plural ("images", "videos").</param>
    public ImageGalleryViewModel(ImageGalleryStore store, IPlatformServices platform, IUiDispatcher dispatcher, Action<GalleryItemViewModel> open, Action<GalleryEntry> reuse,
        string noun = "images")
    {
        _noun = noun;
        _store = store;
        _platform = platform;
        _dispatcher = dispatcher;
        _thumbnails = new ThumbnailCache(dispatcher);
        _open = open;
        _reuse = reuse;
        Models = [AllModels];
        ModelFilter = AllModels;
    }

    /// <summary>The images shown (after the filter), newest first. Replaced as a whole, not changed item by item.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<GalleryItemViewModel> Visible { get; private set; } = [];

    /// <summary>The models of the stored images, for the filter.</summary>
    [ObservableProperty]
    public partial ObservableCollection<string> Models { get; private set; }

    /// <summary>Only images of this model (<see cref="AllModels"/> for all).</summary>
    [ObservableProperty]
    public partial string ModelFilter { get; set; }

    /// <summary>Only images whose prompt (or negative prompt) contains this text.</summary>
    [ObservableProperty]
    public partial string TextFilter { get; set; } = "";

    /// <summary>The image shown large.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSelection))]
    public partial GalleryItemViewModel? SelectedItem { get; set; }

    /// <summary>The images are being read from disk.</summary>
    [ObservableProperty]
    public partial bool IsLoading { get; private set; }

    /// <summary>Whether an image is selected.</summary>
    public bool HasSelection => SelectedItem is not null;

    /// <summary>The selected image waits for the delete confirmation.</summary>
    public bool IsConfirmingDelete => SelectedItem?.IsConfirmingDelete == true;

    /// <summary>The text of the delete button: "Delete", then "Click again to delete".</summary>
    public string DeleteText => IsConfirmingDelete ? "Click again to delete" : "Delete";

    /// <summary>The number of stored images.</summary>
    public int Count => _all.Count;

    /// <summary>There is no stored image.</summary>
    public bool IsEmpty => _all.Count == 0 && !IsLoading;

    /// <summary>"12 of 340 images" under the filter.</summary>
    public string CountText => Visible.Count == _all.Count ? $"{_all.Count} {_noun}" : $"{Visible.Count} of {_all.Count} {_noun}";

    /// <summary>What the strip says while it is empty.</summary>
    public string EmptyText => $"No {_noun} yet: everything you generate is kept here.";

    /// <summary>Whether the folder of the images is shown (the user's disk location, for the tooltip of "open folder").</summary>
    public string Folder => _store.Folder;

    /// <summary>The latest deletion on disk (for tests).</summary>
    internal Task Deleting { get; private set; } = Task.CompletedTask;

    /// <summary>The latest load (for tests).</summary>
    internal Task Loading { get; private set; } = Task.CompletedTask;

    /// <summary>Reads the stored images on a worker thread and shows them.</summary>
    public Task LoadAsync()
    {
        IsLoading = true;
        Loading = Task.Run(() =>
        {
            var entries = _store.Load();
            _dispatcher.Post(() =>
            {
                _all.Clear();
                _all.AddRange(entries.Select(Item));
                IsLoading = false;
                RefreshModels();
                ApplyFilter();
                OnPropertyChanged(nameof(Count));
            });
        });
        return Loading;
    }

    /// <summary>Adds a freshly stored image at the top.</summary>
    /// <param name="entry">The stored entry.</param>
    public void Add(GalleryEntry entry)
    {
        _all.Insert(0, Item(entry));
        RefreshModels();
        ApplyFilter();
        OnPropertyChanged(nameof(Count));
    }

    private GalleryItemViewModel Item(GalleryEntry entry) => new(entry, _store.PathOf(entry), _thumbnails);

    private void RefreshModels()
    {
        var models = _all.Select(item => item.Entry.Model).Where(model => model.Length > 0).Distinct(StringComparer.Ordinal).Order(StringComparer.OrdinalIgnoreCase).ToList();
        string keep = ModelFilter;
        Models = [AllModels, .. models];
        ModelFilter = Models.Contains(keep) ? keep : AllModels;
    }

    partial void OnModelFilterChanged(string value) => ApplyFilter();

    partial void OnTextFilterChanged(string value) => ApplyFilter();

    private void ApplyFilter()
    {
        string text = TextFilter.Trim();
        bool anyModel = ModelFilter is null or AllModels;
        Visible = text.Length == 0 && anyModel
            ? [.. _all]
            : [.. _all.Where(item => (anyModel || item.Entry.Model == ModelFilter)
                && (text.Length == 0 || item.Entry.Prompt.Contains(text, StringComparison.OrdinalIgnoreCase) || item.Entry.NegativePrompt.Contains(text, StringComparison.OrdinalIgnoreCase)))];
        OnPropertyChanged(nameof(CountText));
        OnPropertyChanged(nameof(IsEmpty));
    }

    partial void OnSelectedItemChanged(GalleryItemViewModel? value)
    {
        NotifyDeleteState();
        if (value is not null)
        {
            _open(value);
        }
    }

    private void NotifyDeleteState()
    {
        OnPropertyChanged(nameof(IsConfirmingDelete));
        OnPropertyChanged(nameof(DeleteText));
    }

    /// <summary>Fills the form of the page with the settings of an image (prompt, seed, steps, size, ...).</summary>
    [RelayCommand]
    private void ReuseSettings(GalleryItemViewModel? item)
    {
        if ((item ?? SelectedItem) is { } target)
        {
            _reuse(target.Entry);
        }
    }

    /// <summary>Copies the prompt to the clipboard.</summary>
    [RelayCommand]
    private Task CopyPromptAsync(GalleryItemViewModel? item) => (item ?? SelectedItem) is { } target ? _platform.CopyTextAsync(target.Entry.Prompt) : Task.CompletedTask;

    /// <summary>Shows the image's folder in the file manager.</summary>
    [RelayCommand]
    private void OpenFolder(GalleryItemViewModel? item)
    {
        if ((item ?? SelectedItem) is { } target)
        {
            _platform.Open(_store.DirectoryOf(target.Entry));
        }
    }

    /// <summary>Saves a copy of the image where the user chooses.</summary>
    [RelayCommand]
    private async Task ExportAsync(GalleryItemViewModel? item)
    {
        if ((item ?? SelectedItem) is { } target)
        {
            bool video = target.Entry.Video.Length > 0;
            byte[] bytes = await Task.Run(() => video ? File.ReadAllBytes(_store.PathOfVideo(target.Entry)) : _store.ReadPng(target.Entry));
            string extension = video ? Path.GetExtension(target.Entry.Video) : ".png";
            await Exporter.SaveAsync(_platform, video ? "Save video" : "Save image", $"sharplight-{target.Entry.Seed}{extension}", bytes);
        }
    }

    /// <summary>First click: asks for confirmation; the item shows "Confirm delete".</summary>
    [RelayCommand]
    private void Delete(GalleryItemViewModel? item)
    {
        if ((item ?? SelectedItem) is not { } target)
        {
            return;
        }

        if (!target.IsConfirmingDelete)
        {
            foreach (var other in _all.Where(other => other.IsConfirmingDelete))
            {
                other.IsConfirmingDelete = false;
            }

            target.IsConfirmingDelete = true;
            NotifyDeleteState();
            return;
        }

        _all.Remove(target);
        _thumbnails.Remove(_store.PathOf(target.Entry));
        if (ReferenceEquals(SelectedItem, target))
        {
            SelectedItem = null;
        }

        ApplyFilter();
        OnPropertyChanged(nameof(Count));
        Deleting = Task.Run(() => _store.Delete(target.Entry));
    }

    /// <summary>Withdraws a delete request.</summary>
    [RelayCommand]
    private void CancelDelete(GalleryItemViewModel? item)
    {
        if ((item ?? SelectedItem) is { } target)
        {
            target.IsConfirmingDelete = false;
            NotifyDeleteState();
        }
    }

    /// <summary>Clears both filters.</summary>
    [RelayCommand]
    private void ClearFilters()
    {
        TextFilter = "";
        ModelFilter = AllModels;
    }
}
