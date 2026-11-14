using Avalonia.Threading;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.Platform;

/// <summary><see cref="IUiDispatcher"/> on Avalonia's UI thread.</summary>
internal sealed class AvaloniaDispatcher : IUiDispatcher
{
    /// <inheritdoc/>
    public void Post(Action action)
    {
        if (Dispatcher.UIThread.CheckAccess())
        {
            action();
        }
        else
        {
            Dispatcher.UIThread.Post(action);
        }
    }
}
