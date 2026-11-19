using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Services.Library;

/// <summary>The licence acceptance queue: remembers acceptances in the settings and shows one request at a time.</summary>
internal sealed class LicenseGate : ILicenseGate
{
    private readonly ISettingsService _settings;
    private readonly object _lock = new();
    private readonly List<(LicenseTerms Terms, TaskCompletionSource<bool> Answer)> _queue = [];

    public LicenseGate(ISettingsService settings)
    {
        _settings = settings;
    }

    /// <inheritdoc />
    public event EventHandler? PendingChanged;

    /// <inheritdoc />
    public LicenseTerms? Pending
    {
        get
        {
            lock (_lock)
            {
                return _queue.Count == 0 ? null : _queue[0].Terms;
            }
        }
    }

    /// <inheritdoc />
    public bool IsAccepted(LicenseTerms terms)
    {
        ArgumentNullException.ThrowIfNull(terms);
        return _settings.Current.AcceptedLicenses.Any(accepted => accepted.Key == terms.Key && accepted.TermsVersion == terms.Version);
    }

    /// <inheritdoc />
    public bool HasAccepted(string key) => _settings.Current.AcceptedLicenses.Any(accepted => accepted.Key == key);

    /// <inheritdoc />
    public async Task<bool> RequestAsync(LicenseTerms? terms, CancellationToken cancellationToken = default)
    {
        if (terms is null || IsAccepted(terms))
        {
            return true;
        }

        var answer = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        lock (_lock)
        {
            _queue.Add((terms, answer));
        }

        PendingChanged?.Invoke(this, EventArgs.Empty);
        await using var registration = cancellationToken.Register(() => Withdraw(answer));
        return await answer.Task.ConfigureAwait(false);
    }

    /// <inheritdoc />
    public void Accept()
    {
        LicenseTerms? accepted;
        lock (_lock)
        {
            accepted = _queue.Count == 0 ? null : _queue[0].Terms;
        }

        if (accepted is null)
        {
            return;
        }

        var record = new AcceptedLicense(accepted.Key, accepted.License, accepted.Version, DateTimeOffset.UtcNow);
        _settings.Update(settings => settings with
        {
            AcceptedLicenses = [.. settings.AcceptedLicenses.Where(item => item.Key != record.Key), record],
        });
        List<TaskCompletionSource<bool>> done;
        lock (_lock)
        {
            done = [.. _queue.Where(item => item.Terms.Key == accepted.Key && item.Terms.Version == accepted.Version).Select(item => item.Answer)];
            _queue.RemoveAll(item => item.Terms.Key == accepted.Key && item.Terms.Version == accepted.Version);
        }

        foreach (var answer in done)
        {
            answer.TrySetResult(true);
        }

        PendingChanged?.Invoke(this, EventArgs.Empty);
    }

    /// <inheritdoc />
    public void Decline()
    {
        TaskCompletionSource<bool>? answer;
        lock (_lock)
        {
            if (_queue.Count == 0)
            {
                return;
            }

            answer = _queue[0].Answer;
            _queue.RemoveAt(0);
        }

        answer.TrySetResult(false);
        PendingChanged?.Invoke(this, EventArgs.Empty);
    }

    private void Withdraw(TaskCompletionSource<bool> answer)
    {
        lock (_lock)
        {
            _queue.RemoveAll(item => ReferenceEquals(item.Answer, answer));
        }

        answer.TrySetResult(false);
        PendingChanged?.Invoke(this, EventArgs.Empty);
    }
}
