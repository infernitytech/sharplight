using Sharplight.Studio.Services;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>
/// The optional periodic check of the model feed for newer versions of installed packages (Settings &gt; Models). It runs at start and then
/// when the configured interval has passed; the result is the number the Models item of the navigation shows. The network work runs off the UI thread.
/// </summary>
internal sealed class ModelUpdateWatcher
{
    private static readonly TimeSpan s_poll = TimeSpan.FromMinutes(30);

    private readonly StudioServices _services;
    private readonly Func<Task<int>> _check;
    private readonly TimeProvider _time;
    private CancellationTokenSource? _loop;

    /// <param name="services">The services (settings).</param>
    /// <param name="check">Reads the feed and the library again and returns how many models have a newer version.</param>
    /// <param name="time">The clock.</param>
    public ModelUpdateWatcher(StudioServices services, Func<Task<int>> check, TimeProvider? time = null)
    {
        _services = services;
        _check = check;
        _time = time ?? TimeProvider.System;
    }

    /// <summary>Raised on the UI thread after a check, with the number of models that have a newer version.</summary>
    public event EventHandler<int>? Checked;

    /// <summary>The check is due: it is on in the settings, and none ran for the configured interval.</summary>
    public bool IsDue
    {
        get
        {
            var settings = _services.Settings.Current;
            if (!settings.CheckModelUpdates)
            {
                return false;
            }

            var interval = TimeSpan.FromHours(Math.Clamp(settings.ModelUpdateIntervalHours, 1, 24 * 30));
            return settings.LastModelUpdateCheck is not { } last || _time.GetUtcNow() - last >= interval;
        }
    }

    /// <summary>Checks now when <see cref="IsDue"/>.</summary>
    /// <returns>Whether a check ran.</returns>
    public async Task<bool> CheckIfDueAsync()
    {
        if (!IsDue)
        {
            return false;
        }

        int count = await _check();
        var now = _time.GetUtcNow();
        _services.Settings.Update(current => current with { LastModelUpdateCheck = now });
        Checked?.Invoke(this, count);
        return true;
    }

    /// <summary>Starts checking in the background: once now, then whenever the interval has passed (the setting is read each time, so turning it on or off applies).</summary>
    public void Start()
    {
        if (_loop is not null)
        {
            return;
        }

        _loop = new CancellationTokenSource();
        _ = RunAsync(_loop.Token);
    }

    /// <summary>Stops the background checks.</summary>
    public void Stop()
    {
        _loop?.Cancel();
        _loop = null;
    }

    private async Task RunAsync(CancellationToken cancellationToken)
    {
        try
        {
            using var timer = new PeriodicTimer(s_poll, _time);
            do
            {
                try
                {
                    await CheckIfDueAsync();
                }
                catch (Exception error) when (error is IOException or HttpRequestException or InvalidOperationException)
                {
                    // An unreachable feed is not an error worth showing; the next poll tries again.
                }
            }
            while (await timer.WaitForNextTickAsync(cancellationToken));
        }
        catch (OperationCanceledException)
        {
            // Stopped.
        }
    }
}
