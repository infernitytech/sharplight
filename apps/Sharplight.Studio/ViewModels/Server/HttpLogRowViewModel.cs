using System.Globalization;
using Sharplight.Studio.Services.Server;

namespace Sharplight.Studio.ViewModels.Server;

/// <summary>A row of the request log.</summary>
internal sealed class HttpLogRowViewModel
{
    public HttpLogRowViewModel(HttpLogEntry entry)
    {
        Entry = entry;
    }

    /// <summary>The entry.</summary>
    public HttpLogEntry Entry { get; }

    /// <summary>Local time.</summary>
    public string Time => Entry.Timestamp.ToLocalTime().ToString("HH:mm:ss", CultureInfo.InvariantCulture);

    /// <summary>Method.</summary>
    public string Method => Entry.Method;

    /// <summary>Path.</summary>
    public string Path => Entry.Path;

    /// <summary>Status code.</summary>
    public string Status => Entry.Status.ToString(CultureInfo.InvariantCulture);

    /// <summary>2xx.</summary>
    public bool IsSuccess => Entry.Status is >= 200 and < 300;

    /// <summary>4xx.</summary>
    public bool IsClientError => Entry.Status is >= 400 and < 500;

    /// <summary>5xx.</summary>
    public bool IsServerError => Entry.Status >= 500;

    /// <summary>Duration.</summary>
    public string Duration => Entry.DurationMs >= 1000
        ? (Entry.DurationMs / 1000).ToString("F2", CultureInfo.InvariantCulture) + " s"
        : Entry.DurationMs.ToString("F0", CultureInfo.InvariantCulture) + " ms";

    /// <summary>Client address.</summary>
    public string Client => Entry.Client;
}
