using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Server;

namespace Sharplight.Studio.ViewModels.Server;

/// <summary>A loaded model with its "served" switch.</summary>
internal sealed partial class ServedModelViewModel : ObservableObject
{
    private readonly Action _changed;
    private readonly Func<(bool OpenAi, bool Anthropic)> _apis;

    public ServedModelViewModel(ILoadedModel model, bool served, Action changed, Func<(bool OpenAi, bool Anthropic)>? apis = null)
    {
        Model = model;
        _changed = changed;
        _apis = apis ?? (() => (true, true));
        IsServed = served;
    }

    /// <summary>The model.</summary>
    public ILoadedModel Model { get; }

    /// <summary>Its id (the <c>model</c> field clients send).</summary>
    public string Id => Model.Id;

    /// <summary>Device.</summary>
    public string Device => Model.Device.Name;

    /// <summary>The kind of the model (what it does).</summary>
    public string Kind => Model.Kind;

    /// <summary>The paths of the enabled APIs the model answers on.</summary>
    public IReadOnlyList<string> Endpoints
    {
        get
        {
            var (openAi, anthropic) = _apis();
            return ServedEndpoints.For(Model.Kind, openAi, anthropic);
        }
    }

    /// <summary>The endpoints as a line for the model list, or a note when the engine does not serve this kind over HTTP.</summary>
    public string EndpointsText => Endpoints.Count == 0 ? "not served over HTTP" : string.Join("  ", Endpoints);

    /// <summary>Life-cycle state.</summary>
    public string State => Model.State.ToString();

    /// <summary>Re-reads the model's live state.</summary>
    public void Refresh() => OnPropertyChanged(nameof(State));

    /// <summary>Exposed over HTTP.</summary>
    [ObservableProperty]
    public partial bool IsServed { get; set; }

    partial void OnIsServedChanged(bool value) => _changed();
}
