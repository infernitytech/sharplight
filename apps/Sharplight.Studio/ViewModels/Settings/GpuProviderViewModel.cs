using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>One GPU on the Engine tab: its name and memory, the provider picker (Automatic, Vulkan, CUDA), the provider in use and a note.</summary>
internal sealed partial class GpuProviderViewModel : ObservableObject
{
    private readonly Action<GpuProviderViewModel, ProviderChoice> _chosen;
    private bool _refreshing;

    /// <param name="gpu">The GPU.</param>
    /// <param name="chosen">Called with the new choice when the user picks one.</param>
    public GpuProviderViewModel(PhysicalGpu gpu, Action<GpuProviderViewModel, ProviderChoice> chosen)
    {
        _chosen = chosen;
        Key = gpu.Key;
        Refresh(gpu);
        Selected ??= ProviderChoiceItem.Automatic;
    }

    /// <summary>The GPU's identity.</summary>
    public string Key { get; }

    /// <summary>Device name.</summary>
    [ObservableProperty]
    public partial string Name { get; private set; } = "";

    /// <summary>Memory.</summary>
    [ObservableProperty]
    public partial string MemoryText { get; private set; } = "";

    /// <summary>The picker's entries: Automatic and the providers that can drive the GPU.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<ProviderChoiceItem> Choices { get; private set; } = [];

    /// <summary>The picked entry.</summary>
    [ObservableProperty]
    public partial ProviderChoiceItem? Selected { get; set; }

    /// <summary>Which provider runs the GPU now, e.g. "Running on Vulkan".</summary>
    [ObservableProperty]
    public partial string InUse { get; private set; } = "";

    /// <summary>Why CUDA is not used although it could be, or empty.</summary>
    [ObservableProperty]
    public partial string Note { get; private set; } = "";

    /// <summary>More than one provider can drive the GPU, so the picker is shown.</summary>
    [ObservableProperty]
    public partial bool HasChoice { get; private set; }

    /// <summary>There is a note.</summary>
    public bool HasNote => Note.Length > 0;

    /// <summary>Takes over the GPU's current state (after a choice, the picker's value and the provider in use change).</summary>
    /// <param name="gpu">The GPU as the runtime sees it now.</param>
    public void Refresh(PhysicalGpu gpu)
    {
        _refreshing = true;
        Name = gpu.Name;
        MemoryText = ByteFormat.Format(gpu.TotalMemory);
        HasChoice = gpu.HasChoice;
        Choices = [ProviderChoiceItem.Automatic, .. gpu.Entries.Select(entry => ProviderChoiceItem.For(entry.Provider == ProviderTitles.Cuda ? ProviderChoice.Cuda : ProviderChoice.Vulkan))
            .Distinct().OrderBy(item => item.Value)];
        Selected = Choices.FirstOrDefault(item => item.Value == gpu.Choice) ?? ProviderChoiceItem.Automatic;
        InUse = $"Runs on {ProviderTitles.Of(gpu.Effective.Provider)}" + (gpu.Choice == ProviderChoice.Automatic ? " (automatic)" : "");
        Note = gpu.Note ?? "";
        OnPropertyChanged(nameof(HasNote));
        _refreshing = false;
    }

    partial void OnSelectedChanged(ProviderChoiceItem? value)
    {
        if (!_refreshing && value is not null)
        {
            _chosen(this, value.Value);
        }
    }
}
