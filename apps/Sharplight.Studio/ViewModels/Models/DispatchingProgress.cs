using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>Reports progress on the UI thread.</summary>
internal sealed class DispatchingProgress : IProgress<Services.Hub.PullUpdate>
{
    private readonly IUiDispatcher _dispatcher;
    private readonly Action<Services.Hub.PullUpdate> _handler;

    public DispatchingProgress(IUiDispatcher dispatcher, Action<Services.Hub.PullUpdate> handler)
    {
        _dispatcher = dispatcher;
        _handler = handler;
    }

    /// <inheritdoc/>
    public void Report(Services.Hub.PullUpdate value) => _dispatcher.Post(() => _handler(value));
}
