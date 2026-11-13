namespace Sharplight.Llm.Serving;

/// <summary>Helpers to find marker strings in a text stream that arrives in arbitrary pieces.</summary>
internal static class MarkerScanner
{
    /// <summary>Finds the earliest occurrence of any marker. Returns its index and marker index, or (-1, -1).</summary>
    public static (int Index, int Marker) FindFirst(string text, IReadOnlyList<string> markers, int from = 0)
    {
        int best = -1, bm = -1;
        for (int m = 0; m < markers.Count; m++)
        {
            if (markers[m].Length == 0)
            {
                continue;
            }

            int i = text.IndexOf(markers[m], from, StringComparison.Ordinal);
            if (i >= 0 && (best < 0 || i < best))
            {
                best = i;
                bm = m;
            }
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
            {
                if (string.CompareOrdinal(text, text.Length - len, m, 0, len) == 0)
                {
                    best = len;
                    break;
                }
            }
        }
        return best;
    }
}
