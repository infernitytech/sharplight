using System.Text;

namespace Sharplight.Llm.Serving;

/// <summary>
/// Cuts a text stream at the first stop string. Text that could still turn out to be the start of a stop string is held back
/// until it is disambiguated, so a stop sequence split over several tokens is never partially emitted.
/// </summary>
public sealed class StopStringFilter(IReadOnlyList<string> stops)
{
    private readonly StringBuilder _pending = new();
    private readonly string[] _stops = [.. stops.Where(s => s.Length > 0)];
    public bool Stopped { get; private set; }
    /// <summary>The stop string that ended the stream.</summary>
    public string? Matched { get; private set; }

    /// <summary>Adds text; returns the part that is safe to emit.</summary>
    public string Push(string text)
    {
        if (Stopped || text.Length == 0)
        {
            return "";
        }

        if (_stops.Length == 0)
        {
            return text;
        }

        _pending.Append(text);
        var s = _pending.ToString();
        var (idx, m) = MarkerScanner.FindFirst(s, _stops);
        if (idx >= 0)
        {
            Stopped = true; Matched = _stops[m]; _pending.Clear();
            return s[..idx];
        }
        int hold = MarkerScanner.PartialSuffix(s, _stops);
        _pending.Clear().Append(s, s.Length - hold, hold);
        return s[..(s.Length - hold)];
    }

    /// <summary>End of stream without a match: releases the held-back text.</summary>
    public string Flush()
    {
        var s = _pending.ToString();
        _pending.Clear();
        return s;
    }
}
