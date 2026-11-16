using Sharplight.Studio.Converters;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Threading;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Image;

/// <summary>
/// The step text and live preview of a running generation, reported by the run's worker and shown on the UI thread at most about 15
/// times a second (<see cref="UpdateThrottle"/>). Previews are encoded and decoded into bitmaps on the worker, so the UI thread only
/// swaps them in; a preview superseded before it was shown is not encoded at all.
/// </summary>
internal sealed class StepDisplay : IDisposable
{
    private readonly IUiDispatcher _dispatcher;
    private readonly Action<string, byte[]?> _show;
    private readonly UpdateThrottle _throttle;
    private readonly Lock _lock = new();
    private string _step = "";
    private ImageInput? _preview;

    /// <param name="dispatcher">Runs <paramref name="show"/> on the UI thread.</param>
    /// <param name="show">Shows a step text and, when not null, a new preview PNG (on the UI thread).</param>
    public StepDisplay(IUiDispatcher dispatcher, Action<string, byte[]?> show)
    {
        _dispatcher = dispatcher;
        _show = show;
        _throttle = new UpdateThrottle(Fire);
    }

    /// <summary>Records a step (any thread).</summary>
    /// <param name="step">The step text.</param>
    /// <param name="preview">The step's preview, or null.</param>
    public void Report(string step, ImageInput? preview)
    {
        lock (_lock)
        {
            _step = step;
            _preview = preview ?? _preview;
        }

        _throttle.Request();
    }

    /// <summary>Shows the last step now (when the run finished).</summary>
    public void Flush() => _throttle.Flush();

    /// <summary>Drops a step not shown yet.</summary>
    public void Dispose() => _throttle.Dispose();

    private void Fire()
    {
        string step;
        ImageInput? preview;
        lock (_lock)
        {
            step = _step;
            preview = _preview;
            _preview = null;
        }

        byte[]? png = null;
        if (preview is not null)
        {
            png = ImageFiles.ToPng(preview);
            DecodedBitmaps.Prepare(png);
        }

        _dispatcher.Post(() => _show(step, png));
    }
}
