using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels;

/// <summary>A loaded model as offered in pickers.</summary>
internal sealed class ModelOptionViewModel
{
    public ModelOptionViewModel(ILoadedModel model)
    {
        Model = model;
    }

    /// <summary>The model.</summary>
    public ILoadedModel Model { get; }

    /// <summary>Its id.</summary>
    public string Id => Model.Id;

    /// <summary>Device and context, for the picker's second line.</summary>
    public string Detail => $"{Model.Device.Name} · {ByteFormat.Count(Model.ContextLength)} ctx";

    /// <summary>Badges of what it can do.</summary>
    public CapabilityBadges Badges => new(Model.Capabilities);
}
