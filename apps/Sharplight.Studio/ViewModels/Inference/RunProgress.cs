namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>
/// The latest progress a run reported (from its worker thread), applied to its <see cref="RunItemViewModel"/> on the UI thread when the
/// throttle lets it through: reports in between are superseded, not queued.
/// </summary>
internal sealed class RunProgress
{
    private readonly Lock _lock = new();
    private double? _fraction;
    private string? _text;
    private bool _changed;

    /// <summary>Records a report: a fraction (null: indeterminate) and, when not null, a new detail text.</summary>
    /// <param name="fraction">The fraction done, 0..1, or null when it cannot be measured.</param>
    /// <param name="text">The new detail text, or null to keep the last one.</param>
    public void Set(double? fraction, string? text)
    {
        lock (_lock)
        {
            _fraction = fraction;
            _text = text ?? _text;
            _changed = true;
        }
    }

    /// <summary>Shows the latest report on <paramref name="item"/> (on the UI thread).</summary>
    /// <param name="item">The run.</param>
    public void ApplyTo(RunItemViewModel item)
    {
        double? fraction;
        string? text;
        lock (_lock)
        {
            if (!_changed)
            {
                return;
            }

            _changed = false;
            fraction = _fraction;
            text = _text;
        }

        if (!item.IsActive)
        {
            return;
        }

        if (fraction is { } value)
        {
            item.IsIndeterminate = false;
            item.Progress = Math.Clamp(value, 0, 1);
        }
        else
        {
            item.IsIndeterminate = true;
        }

        if (text is not null)
        {
            item.Detail = text;
        }
    }
}
