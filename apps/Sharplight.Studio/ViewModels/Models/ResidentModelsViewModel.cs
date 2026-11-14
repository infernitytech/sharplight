using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>The resident-models panel: every model kept on a device, its state and memory, and each device's budget.</summary>
internal sealed partial class ResidentModelsViewModel : ObservableObject
{
    private readonly StudioServices _services;
    private readonly INavigator _navigator;

    public ResidentModelsViewModel(StudioServices services, INavigator navigator)
    {
        _services = services;
        _navigator = navigator;
        services.Runtime.ModelsChanged += (_, _) => services.Dispatcher.Post(Refresh);
        Refresh();
    }

    /// <summary>The models.</summary>
    public ObservableCollection<LoadedModelViewModel> Models { get; } = [];

    /// <summary>Device memory.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<DeviceBudgetViewModel> Devices { get; set; } = [];

    /// <summary>Status or error.</summary>
    [ObservableProperty]
    public partial string Status { get; set; } = "";

    /// <summary>True when no model is resident.</summary>
    public bool IsEmpty => Models.Count == 0;

    /// <summary>Policy summary, e.g. <c>Idle models unload after 15 min · on-demand loading on</c>.</summary>
    public string PolicyText
    {
        get
        {
            var residency = _services.Settings.Current.Residency;
            var parts = new List<string> { $"{residency.ReserveMiB} MiB kept free per device" };
            parts.Add(residency.IdleUnloadMinutes > 0 ? $"idle models unload after {residency.IdleUnloadMinutes} min" : "no idle unloading");
            if (residency.MaxLoadedModels > 0)
            {
                parts.Add($"at most {residency.MaxLoadedModels} resident");
            }

            parts.Add(residency.LoadOnDemand ? "library models load on demand" : "on-demand loading off");
            return string.Join(" · ", parts);
        }
    }

    /// <summary>Re-reads models and devices.</summary>
    public void Refresh()
    {
        var current = _services.Runtime.LoadedModels;
        if (Models.Select(item => item.Model).SequenceEqual(current))
        {
            foreach (var item in Models)
            {
                item.Refresh();
            }
        }
        else
        {
            Models.Clear();
            foreach (var model in current)
            {
                Models.Add(new LoadedModelViewModel(model));
            }

            OnPropertyChanged(nameof(IsEmpty));
        }

        long reserve = Math.Max(0, _services.Settings.Current.Residency.ReserveMiB) * (1L << 20);
        Devices = [.. _services.Runtime.GetDeviceUsage().Select(usage =>
            new DeviceBudgetViewModel(usage, [.. current.Where(model => model.Device.Id == usage.Id)], reserve))];
        OnPropertyChanged(nameof(PolicyText));
    }

    [RelayCommand]
    private async Task UnloadAsync(LoadedModelViewModel model)
    {
        model.IsUnloading = true;
        Status = $"Unloading {model.Id}…";
        try
        {
            await _services.Runtime.UnloadAsync(model.Id);
            Status = $"Unloaded {model.Id}.";
        }
        catch (Exception error) when (error is InvalidOperationException or TimeoutException)
        {
            Status = error.Message;
            model.IsUnloading = false;
        }
    }

    [RelayCommand]
    private void ChatWith(LoadedModelViewModel model) => _navigator.OpenChat(model.Model);

    [RelayCommand]
    private void OpenLibrary() => _navigator.OpenLibrary();
}
