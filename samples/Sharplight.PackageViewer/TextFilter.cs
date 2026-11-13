namespace Sharplight.PackageViewer;

/// <summary>
/// The filter boxes of every table: the filter text is split at whitespace and a row matches when every term occurs
/// (case-insensitively) in at least one of its fields.
/// </summary>
internal static class TextFilter
{
    public static string[] Terms(string? filter) =>
        string.IsNullOrWhiteSpace(filter) ? [] : filter.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

    public static bool Matches(string[] terms, params ReadOnlySpan<string> fields)
    {
        foreach (var term in terms)
        {
            if (!MatchesTerm(term, fields))
            {
                return false;
            }
        }

        return true;
    }

    public static bool MatchesTerm(string term, params ReadOnlySpan<string> fields)
    {
        foreach (var field in fields)
        {
            if (field.Contains(term, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>The rows matching <paramref name="filter"/>; the input list itself when the filter is empty.</summary>
    public static IReadOnlyList<T> Apply<T>(IReadOnlyList<T> rows, string? filter)
        where T : IFilterableRow
    {
        var terms = Terms(filter);
        if (terms.Length == 0)
        {
            return rows;
        }

        var result = new List<T>();
        foreach (var row in rows)
        {
            if (row.Matches(terms))
            {
                result.Add(row);
            }
        }

        return result;
    }
}
