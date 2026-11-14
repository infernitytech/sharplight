namespace Sharplight.Studio.Services.Platform;

/// <summary>Runs work on the UI thread.</summary>
internal interface IUiDispatcher
{
    /// <summary>Queues <paramref name="action"/> on the UI thread (runs it inline when already there).</summary>
    public void Post(Action action);
}
