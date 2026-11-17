using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Image;

/// <summary>
/// The LoRA slots of an image model (the load panel and the Image page share it): add from a file or from the library's LoRA files,
/// remove, set the strength of each. Changes raise <see cref="Changed"/>.
/// </summary>
internal sealed partial class LoraListViewModel : ObservableObject
{
    private readonly IPlatformServices _platform;
    private readonly Func<IReadOnlyList<LibraryEntry>> _library;

    /// <param name="platform">The file picker.</param>
    /// <param name="library">The library entries; the LoRA files among them are offered.</param>
    public LoraListViewModel(IPlatformServices platform, Func<IReadOnlyList<LibraryEntry>> library)
    {
        _platform = platform;
        _library = library;
        Slots.CollectionChanged += (_, _) =>
        {
            OnPropertyChanged(nameof(HasSlots));
            Changed?.Invoke(this, EventArgs.Empty);
        };
    }

    /// <summary>Raised when a LoRA was added or removed or a strength changed.</summary>
    public event EventHandler? Changed;

    /// <summary>The LoRAs.</summary>
    public ObservableCollection<LoraSlotViewModel> Slots { get; } = [];

    /// <summary>There is at least one LoRA.</summary>
    public bool HasSlots => Slots.Count > 0;

    /// <summary>The LoRA files of the library that are not in a slot yet.</summary>
    public IReadOnlyList<LibraryEntry> Available
        => [.. _library().Where(entry => entry.Kind == ModelKindCatalog.Lora && Slots.All(slot => slot.Path != entry.Path))];

    /// <summary>Tells the view that the library changed, so the LoRA files it offers are listed again.</summary>
    public void RefreshAvailable() => OnPropertyChanged(nameof(Available));

    /// <summary>The adapters, in slot order.</summary>
    public IReadOnlyList<ModelLora> Adapters => [.. Slots.Select(slot => slot.ToModelLora())];

    /// <summary>Replaces the slots with <paramref name="adapters"/>.</summary>
    /// <param name="adapters">The adapters to show.</param>
    public void Set(IEnumerable<ModelLora> adapters)
    {
        foreach (var slot in Slots)
        {
            slot.PropertyChanged -= OnSlotChanged;
        }

        Slots.Clear();
        foreach (var adapter in adapters)
        {
            Add(adapter.Path, adapter.Scale);
        }
    }

    /// <summary>Adds a LoRA unless its file is in a slot already.</summary>
    /// <param name="path">The file.</param>
    /// <param name="scale">The strength.</param>
    public void Add(string path, double scale = 1)
    {
        if (Slots.Any(slot => slot.Path == path))
        {
            return;
        }

        var added = new LoraSlotViewModel(path, scale, Remove);
        added.PropertyChanged += OnSlotChanged;
        Slots.Add(added);
        OnPropertyChanged(nameof(Available));
    }

    private void Remove(LoraSlotViewModel slot)
    {
        slot.PropertyChanged -= OnSlotChanged;
        Slots.Remove(slot);
        OnPropertyChanged(nameof(Available));
    }

    private void OnSlotChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs args)
    {
        if (args.PropertyName == nameof(LoraSlotViewModel.Scale))
        {
            Changed?.Invoke(this, EventArgs.Empty);
        }
    }

    /// <summary>Whether the slots hold other adapters (files or strengths) than <paramref name="adapters"/>.</summary>
    /// <param name="adapters">The adapters to compare with.</param>
    public bool Differs(IReadOnlyList<ModelLora> adapters) => !Adapters.SequenceEqual(adapters);

    [RelayCommand]
    private async Task AddFileAsync()
    {
        var files = await _platform.PickFilesAsync("Choose a LoRA file", [new FileFilter("LoRA files", ["*.safetensors"])], allowMultiple: true);
        foreach (string file in files)
        {
            Add(file);
        }
    }

    [RelayCommand]
    private void AddLibrary(LibraryEntry entry) => Add(entry.Path);
}
