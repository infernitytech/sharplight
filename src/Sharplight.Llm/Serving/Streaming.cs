using System.Text;

namespace Sharplight.Llm.Serving;

/// <summary>Helpers to find marker strings in a text stream that arrives in arbitrary pieces.</summary>
public static class MarkerScanner
{
    /// <summary>Finds the earliest occurrence of any marker. Returns its index and marker index, or (-1, -1).</summary>
    public static (int Index, int Marker) FindFirst(string text, IReadOnlyList<string> markers, int from = 0)
    {
        int best = -1, bm = -1;
        for (int m = 0; m < markers.Count; m++)
        {
            if (markers[m].Length == 0) continue;
            int i = text.IndexOf(markers[m], from, StringComparison.Ordinal);
            if (i >= 0 && (best < 0 || i < best)) { best = i; bm = m; }
        }
        return (best, bm);
    }

    /// <summary>Length of the longest suffix of <paramref name="text"/> that is a proper prefix of some marker (text that might still become a marker).</summary>
    public static int PartialSuffix(string text, IReadOnlyList<string> markers)
    {
        int best = 0;
        foreach (var m in markers)
        {
            int max = Math.Min(m.Length - 1, text.Length);
            for (int len = max; len > best; len--)
                if (string.CompareOrdinal(text, text.Length - len, m, 0, len) == 0) { best = len; break; }
        }
        return best;
    }
}

/// <summary>
/// Cuts a text stream at the first stop string. Text that could still turn out to be the start of a stop string is held back
/// until it is disambiguated, so a stop sequence split over several tokens is never partially emitted.
/// </summary>
public sealed class StopStringFilter(IReadOnlyList<string> stops)
{
    readonly StringBuilder _pending = new();
    readonly string[] _stops = [.. stops.Where(s => s.Length > 0)];
    public bool Stopped { get; private set; }
    /// <summary>The stop string that ended the stream.</summary>
    public string? Matched { get; private set; }

    /// <summary>Adds text; returns the part that is safe to emit.</summary>
    public string Push(string text)
    {
        if (Stopped || text.Length == 0) return "";
        if (_stops.Length == 0) return text;
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
