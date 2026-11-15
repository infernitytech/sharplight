using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>
/// A parameter panel bound to a task options type: <see cref="Options"/> maps the current values to it (for example
/// <c>SpeechRecognitionOptions</c>), so the page passes exactly what the panel shows to the task client.
/// </summary>
/// <typeparam name="TOptions">The task options type.</typeparam>
internal sealed class ParameterPanelViewModel<TOptions> : ParameterPanelViewModel
{
    private readonly Func<ParameterPanelViewModel, TOptions> _map;

    /// <param name="storage">The page storage.</param>
    /// <param name="pageId">The page's id.</param>
    /// <param name="parameters">The parameters, in display order.</param>
    /// <param name="map">Builds the options from the panel's values.</param>
    /// <param name="builtIn">Presets that ship with the page.</param>
    public ParameterPanelViewModel(IPageStorage storage, string pageId, IReadOnlyList<ParameterViewModel> parameters, Func<ParameterPanelViewModel, TOptions> map,
        IReadOnlyList<ParameterPreset>? builtIn = null)
        : base(storage, pageId, parameters, builtIn)
    {
        _map = map;
    }

    /// <summary>The task options the current values describe.</summary>
    public TOptions Options => _map(this);
}
