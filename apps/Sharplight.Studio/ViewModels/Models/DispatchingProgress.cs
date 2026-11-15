using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>Reports progress on the UI thread.</summary>
/// <typeparam name="T">The progress value.</typeparam>
internal sealed class DispatchingProgress<T> : IProgress<T>
{
    private readonly IUiDispatcher _dispatcher;
    private readonly Action<T> _handler;

    public DispatchingProgress(IUiDispatcher dispatcher, Action<T> handler)
    {
        _dispatcher = dispatcher;
        _handler = handler;
    }

    /// <inheritdoc/>
    public void Report(T value) => _dispatcher.Post(() => _handler(value));
}
