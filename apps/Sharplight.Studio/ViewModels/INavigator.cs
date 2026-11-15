using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels;

/// <summary>Moves between pages (implemented by the main window's view model).</summary>
internal interface INavigator
{
    /// <summary>Shows the chat, selecting <paramref name="model"/> when given.</summary>
    public void OpenChat(ILoadedModel? model);

    /// <summary>Shows the page of a model kind (the chat for text generation, the transcribe and speak pages, ...), selecting <paramref name="model"/> when given.</summary>
    /// <param name="kind">The kind (<see cref="Sharplight.Models.ModelKinds"/>).</param>
    /// <param name="model">The model to choose, or null.</param>
    /// <returns>Whether the kind has a page; when false nothing changed.</returns>
    public bool OpenKind(string kind, ILoadedModel? model);

    /// <summary>Shows the model library.</summary>
    public void OpenLibrary();

    /// <summary>Shows the Hugging Face search.</summary>
    public void OpenDiscover();

    /// <summary>Shows the Hugging Face search filtered to the models of <paramref name="task"/>.</summary>
    /// <param name="task">The model type to browse.</param>
    public void OpenDiscover(Sharplight.Hub.ModelTask task);

    /// <summary>Shows the model library filtered to one model kind.</summary>
    /// <param name="kind">The kind (<see cref="Sharplight.Models.ModelKinds"/>).</param>
    public void OpenLibrary(string kind);
}
