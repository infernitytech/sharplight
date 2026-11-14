using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Server;

/// <summary>A loaded model with its "served" switch.</summary>
internal sealed partial class ServedModelViewModel : ObservableObject
{
    private readonly Action _changed;

    public ServedModelViewModel(ILoadedModel model, bool served, Action changed)
    {
        Model = model;
        _changed = changed;
        IsServed = served;
    }

    /// <summary>The model.</summary>
    public ILoadedModel Model { get; }

    /// <summary>Its id (the <c>model</c> field clients send).</summary>
    public string Id => Model.Id;

    /// <summary>Device.</summary>
    public string Device => Model.Device.Name;

    /// <summary>Life-cycle state.</summary>
    public string State => Model.State.ToString();

    /// <summary>Re-reads the model's live state.</summary>
    public void Refresh() => OnPropertyChanged(nameof(State));

    /// <summary>Exposed over HTTP.</summary>
    [ObservableProperty]
    public partial bool IsServed { get; set; }

    partial void OnIsServedChanged(bool value) => _changed();
}
