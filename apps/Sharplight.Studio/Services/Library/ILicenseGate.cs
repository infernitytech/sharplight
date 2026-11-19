namespace Sharplight.Studio.Services.Library;

/// <summary>Asks the user to accept licence terms before a download; the answer is remembered per model and version of the terms.</summary>
internal interface ILicenseGate
{
    /// <summary>Raised when <see cref="Pending"/> changed (from any thread).</summary>
    public event EventHandler? PendingChanged;

    /// <summary>The terms the dialog shows now (the oldest unanswered request), or null.</summary>
    public LicenseTerms? Pending { get; }

    /// <summary>Whether the user accepted exactly these terms before.</summary>
    /// <param name="terms">The terms.</param>
    public bool IsAccepted(LicenseTerms terms);

    /// <summary>Whether the user accepted any terms of the model <paramref name="key"/> before.</summary>
    /// <param name="key">The acceptance key.</param>
    public bool HasAccepted(string key);

    /// <summary>Completes with true at once when the terms were accepted before; otherwise queues the dialog and completes with the answer (false when cancelled).</summary>
    /// <param name="terms">The terms, or null when the model needs none (completes with true).</param>
    /// <param name="cancellationToken">Withdraws the request (completes with false).</param>
    public Task<bool> RequestAsync(LicenseTerms? terms, CancellationToken cancellationToken = default);

    /// <summary>The user accepted the pending terms: remembered, and the waiting requests for the same model complete.</summary>
    public void Accept();

    /// <summary>The user declined the pending terms.</summary>
    public void Decline();
}
