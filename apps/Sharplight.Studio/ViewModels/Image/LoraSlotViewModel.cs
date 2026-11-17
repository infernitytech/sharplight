using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Image;

/// <summary>One LoRA of an image model: its file and the strength it is merged with.</summary>
internal sealed partial class LoraSlotViewModel : ObservableObject
{
    private readonly Action<LoraSlotViewModel> _remove;

    /// <param name="path">The LoRA file.</param>
    /// <param name="scale">The strength.</param>
    /// <param name="remove">Removes the slot from its list.</param>
    public LoraSlotViewModel(string path, double scale, Action<LoraSlotViewModel> remove)
    {
        Path = path;
        Scale = scale;
        _remove = remove;
    }

    /// <summary>The LoRA file.</summary>
    public string Path { get; }

    /// <summary>The file name without extension.</summary>
    public string Name => System.IO.Path.GetFileNameWithoutExtension(Path);

    /// <summary>The strength: 1 is as trained, 0 has no effect, negative values invert the adapter.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ScaleText))]
    public partial double Scale { get; set; }

    /// <summary>The strength as text.</summary>
    public string ScaleText => Scale.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>The adapter the slot describes.</summary>
    public ModelLora ToModelLora() => new(Path, (float)Math.Round(Scale, 3));

    [RelayCommand]
    private void Remove() => _remove(this);
}
