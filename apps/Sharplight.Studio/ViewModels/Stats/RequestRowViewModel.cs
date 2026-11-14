using System.Globalization;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Stats;

/// <summary>A recent request.</summary>
internal sealed class RequestRowViewModel
{
    public RequestRowViewModel(RequestRecord record)
    {
        Record = record;
    }

    /// <summary>The record.</summary>
    public RequestRecord Record { get; }

    /// <summary>Local time.</summary>
    public string Time => Record.Timestamp.ToLocalTime().ToString("dd MMM HH:mm:ss", CultureInfo.InvariantCulture);

    /// <summary>Model.</summary>
    public string Model => Record.Model;

    /// <summary>Chat or API.</summary>
    public string Source => Record.Source == RequestSource.Chat ? "Chat" : "API";

    /// <summary>Prompt / completion tokens.</summary>
    public string Tokens => $"{Record.Stats.PromptTokens} → {Record.Stats.CompletionTokens}";

    /// <summary>Decode speed.</summary>
    public string Decode => Record.Stats.DecodeTokensPerSecond.ToString("F1", CultureInfo.InvariantCulture);

    /// <summary>Prefill speed.</summary>
    public string Prefill => Record.Stats.PrefillTokensPerSecond.ToString("F0", CultureInfo.InvariantCulture);

    /// <summary>Time to first token.</summary>
    public string Ttft => Record.Stats.TimeToFirstTokenMs.ToString("F0", CultureInfo.InvariantCulture) + " ms";

    /// <summary>Draft acceptance.</summary>
    public string Acceptance => Record.Stats.AcceptanceRate is { } rate ? rate.ToString("P0", CultureInfo.InvariantCulture) : "–";

    /// <summary>Finish reason.</summary>
    public string Finish => Record.Stats.Finish;
}
