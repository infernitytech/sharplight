using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A navigator that records where it was asked to go.</summary>
internal sealed class FakeNavigator : INavigator
{
    public List<string> Visited { get; } = [];

    public void OpenChat(ILoadedModel? model) => Visited.Add("chat");

    /// <summary>The kinds that have a page.</summary>
    public HashSet<string> Pages { get; } = [Sharplight.Models.ModelKinds.SpeechRecognition, Sharplight.Models.ModelKinds.SpeechSynthesis, Sharplight.Models.ModelKinds.ImageGeneration];

    public bool OpenKind(string kind, ILoadedModel? model)
    {
        if (kind == Sharplight.Models.ModelKinds.TextGeneration)
        {
            OpenChat(model);
            return true;
        }

        if (!Pages.Contains(kind))
        {
            return false;
        }

        Visited.Add($"page:{kind}");
        return true;
    }

    public void OpenLibrary() => Visited.Add("library");

    public void OpenDiscover() => Visited.Add("discover");

    public void OpenDiscover(Sharplight.Hub.ModelTask task) => Visited.Add($"discover:{task}");

    public void OpenLibrary(string kind) => Visited.Add($"library:{kind}");
}
