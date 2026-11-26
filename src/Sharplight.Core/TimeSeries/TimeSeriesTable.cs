using System.Globalization;
using System.Text;

namespace Sharplight.TimeSeries;

/// <summary>
/// A table of series read from delimited text (CSV, semicolon- or tab-separated): an optional header row, an optional time column (the first column when
/// its cells are not numbers) and the numeric columns, each a series for <see cref="ITimeSeriesForecaster"/>. Empty or non-numeric cells of a numeric column
/// become missing values (<see cref="float.NaN"/>); quoted cells may hold the delimiter and doubled quotes.
/// </summary>
public sealed class TimeSeriesTable
{
    private TimeSeriesTable(IReadOnlyList<string>? timestamps, string? timeColumn, IReadOnlyList<TimeSeriesColumn> columns)
    {
        Timestamps = timestamps;
        TimeColumn = timeColumn;
        Columns = columns;
    }

    /// <summary>The cells of the time column, one per row, or <see langword="null"/> when the table has none.</summary>
    public IReadOnlyList<string>? Timestamps { get; }

    /// <summary>The header of the time column, or <see langword="null"/>.</summary>
    public string? TimeColumn { get; }

    /// <summary>The numeric columns.</summary>
    public IReadOnlyList<TimeSeriesColumn> Columns { get; }

    /// <summary>The data rows.</summary>
    public int RowCount => Columns.Count > 0 ? Columns[0].Values.Length : Timestamps?.Count ?? 0;

    /// <summary>The column named <paramref name="name"/> (ordinal, case-insensitive).</summary>
    /// <exception cref="KeyNotFoundException">There is no such numeric column.</exception>
    public TimeSeriesColumn Column(string name)
        => Columns.FirstOrDefault(c => string.Equals(c.Name, name, StringComparison.OrdinalIgnoreCase)) ?? throw new KeyNotFoundException($"No numeric column '{name}'.");

    /// <summary>Reads a file.</summary>
    /// <param name="path">The file.</param>
    public static TimeSeriesTable Load(string path)
    {
        using var reader = new StreamReader(path);
        return Read(reader);
    }

    /// <summary>Parses delimited text.</summary>
    /// <param name="text">The text.</param>
    public static TimeSeriesTable Parse(string text)
    {
        using var reader = new StringReader(text);
        return Read(reader);
    }

    /// <summary>Reads delimited text. The delimiter is the first of tab, semicolon and comma found in the first line.</summary>
    /// <param name="reader">The text.</param>
    /// <exception cref="FormatException">The text has no rows or no numeric column.</exception>
    public static TimeSeriesTable Read(TextReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        var lines = new List<string>();
        while (reader.ReadLine() is { } line)
        {
            if (line.Trim().Length > 0)
            {
                lines.Add(line);
            }
        }

        if (lines.Count == 0)
        {
            throw new FormatException("The table has no rows.");
        }

        char delimiter = lines[0].Contains('\t') ? '\t' : lines[0].Contains(';') ? ';' : ',';
        var rows = lines.Select(line => Split(line, delimiter)).ToList();
        int width = rows.Max(r => r.Count);
        bool header = rows[0].Any(cell => cell.Length > 0 && !IsNumber(cell)) && rows.Count > 1 && rows.Skip(1).Any(r => r.Any(IsNumber));
        var names = header ? rows[0] : [];
        var data = header ? rows.Skip(1).ToList() : rows;
        string Cell(List<string> row, int column) => column < row.Count ? row[column] : "";

        // A column is numeric when most of its non-empty cells parse; the first non-numeric column with distinct cells is the time column.
        var numeric = new List<int>();
        int? time = null;
        for (int c = 0; c < width; c++)
        {
            var cells = data.Select(r => Cell(r, c)).Where(cell => cell.Length > 0).ToList();
            int numbers = cells.Count(IsNumber);
            if (cells.Count > 0 && numbers * 2 > cells.Count && !(c == 0 && LooksLikeTime(cells)))
            {
                numeric.Add(c);
            }
            else if (time is null && cells.Count > 0)
            {
                time = c;
            }
        }

        if (numeric.Count == 0 && time == 0 && data.All(r => IsNumber(Cell(r, 0)) || Cell(r, 0).Length == 0))
        {
            numeric.Add(0);   // a single steadily increasing column is the series itself
            time = null;
        }

        if (numeric.Count == 0)
        {
            throw new FormatException("The table has no numeric column.");
        }

        var columns = numeric.Select(c => new TimeSeriesColumn(
            header && c < names.Count && names[c].Length > 0 ? names[c] : $"column {c + 1}",
            [.. data.Select(r => float.TryParse(Cell(r, c), NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ? v : float.NaN)])).ToList();
        IReadOnlyList<string>? timestamps = time is { } t ? [.. data.Select(r => Cell(r, t))] : null;
        string? timeName = time is { } tn && header && tn < names.Count ? names[tn] : null;
        return new TimeSeriesTable(timestamps, timeName, columns);
    }

    private static bool IsNumber(string cell) => float.TryParse(cell, NumberStyles.Float, CultureInfo.InvariantCulture, out _);

    // Integer-like first columns are row indices or years, not values, when they increase steadily (0, 1, 2, ...; 1990, 1991, ...).
    private static bool LooksLikeTime(List<string> cells)
    {
        if (cells.Count < 3 || !cells.All(cell => long.TryParse(cell, NumberStyles.Integer, CultureInfo.InvariantCulture, out _)))
        {
            return false;
        }

        var values = cells.Select(cell => long.Parse(cell, CultureInfo.InvariantCulture)).ToList();
        long step = values[1] - values[0];
        return step > 0 && values.Zip(values.Skip(1)).All(pair => pair.Second - pair.First == step);
    }

    private static List<string> Split(string line, char delimiter)
    {
        var cells = new List<string>();
        var cell = new StringBuilder();
        bool quoted = false;
        for (int i = 0; i < line.Length; i++)
        {
            char ch = line[i];
            if (quoted)
            {
                if (ch == '"' && i + 1 < line.Length && line[i + 1] == '"')
                {
                    cell.Append('"');
                    i++;
                }
                else if (ch == '"')
                {
                    quoted = false;
                }
                else
                {
                    cell.Append(ch);
                }
            }
            else if (ch == '"')
            {
                quoted = true;
            }
            else if (ch == delimiter)
            {
                cells.Add(cell.ToString().Trim());
                cell.Clear();
            }
            else
            {
                cell.Append(ch);
            }
        }

        cells.Add(cell.ToString().Trim());
        return cells;
    }
}
