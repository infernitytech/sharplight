using System.Text;
using Sharplight.Hub.Catalog;

namespace Sharplight.CatalogTool;

/// <summary>
/// Generates the model tables of docs/compatible-models.md from the catalog. A table sits between a begin marker naming its
/// section (<c>&lt;!-- verified-catalog:table 1.1 ... --&gt;</c>) and <see cref="EndMarker"/>; everything outside the markers (the
/// prose) is kept as written.
/// </summary>
internal static class CompatibleModelsDocument
{
    /// <summary>Starts a generated table; the section id follows.</summary>
    public const string BeginPrefix = "<!-- verified-catalog:table ";

    /// <summary>Ends a generated table.</summary>
    public const string EndMarker = "<!-- verified-catalog:end -->";

    /// <summary><paramref name="markdown"/> with every generated table replaced by the catalog's.</summary>
    /// <param name="markdown">The document.</param>
    /// <param name="catalog">The catalog.</param>
    /// <exception cref="InvalidDataException">A marker is unbalanced or names an unknown section.</exception>
    public static string Render(string markdown, VerifiedCatalog catalog)
    {
        var lines = markdown.ReplaceLineEndings("\n").Split('\n');
        var output = new StringBuilder();
        for (int i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            output.Append(line);
            if (i < lines.Length - 1)
            {
                output.Append('\n');
            }

            if (!line.StartsWith(BeginPrefix, StringComparison.Ordinal))
            {
                continue;
            }

            var id = line[BeginPrefix.Length..].Split(' ', 2)[0];
            var section = catalog.DocSections.FirstOrDefault(candidate => candidate.Id == id)
                ?? throw new InvalidDataException($"The document names the unknown catalog section '{id}' (line {i + 1}).");
            int end = Array.FindIndex(lines, i + 1, candidate => candidate == EndMarker);
            if (end < 0)
            {
                throw new InvalidDataException($"The catalog table of section {id} (line {i + 1}) has no end marker.");
            }

            output.Append(RenderTable(section, catalog)).Append(EndMarker);
            if (end < lines.Length - 1)
            {
                output.Append('\n');
            }

            i = end;
        }

        return output.ToString();
    }

    /// <summary>The table of <paramref name="section"/>: header, separator and one line per row, each ending with a line break.</summary>
    public static string RenderTable(CatalogDocSection section, VerifiedCatalog catalog)
    {
        var table = new StringBuilder();
        AppendRow(table, section.Columns.Select(column => column.Title));
        table.Append('|').AppendJoin("", section.Columns.Select(_ => "---|")).Append('\n');
        foreach (var entry in catalog.Entries)
        {
            foreach (var row in entry.Doc.Where(row => row.Section == section.Id))
            {
                AppendRow(table, section.Columns.Select(column => Escape(Cell(entry, row, column.Key))));
            }
        }

        return table.ToString();
    }

    /// <summary>The text of a cell: the row's own, else the value derived from the entry.</summary>
    public static string Cell(CatalogEntry entry, CatalogDocRow row, string key)
    {
        if (row.Cells.TryGetValue(key, out var text))
        {
            return text;
        }

        return key switch
        {
            "model" => entry.Name,
            "repositories" => string.Join(", ", entry.Variants.Select(variant => variant.Repository).Concat(entry.Companions.Select(companion => companion.Repository)).Distinct()),
            "formats" => string.Join(", ", entry.Variants.Select(variant => variant.Label)),
            "providers" => string.Join(", ", entry.Verification.Providers),
            "verification" => entry.Verification.Summary,
            "notes" => entry.Notes ?? "",
            "date" => entry.Verification.Date,
            "kind" => entry.Kind,
            _ => "",
        };
    }

    /// <summary>Appends <c>| a | b |</c> (an empty cell as <c>| |</c>) and a line break.</summary>
    private static void AppendRow(StringBuilder table, IEnumerable<string> cells)
    {
        table.Append('|');
        foreach (var cell in cells)
        {
            table.Append(cell.Length == 0 ? " " : $" {cell} ").Append('|');
        }

        table.Append('\n');
    }

    private static string Escape(string text) => text.Replace("\n", " ", StringComparison.Ordinal).Replace("|", "\\|", StringComparison.Ordinal);
}
