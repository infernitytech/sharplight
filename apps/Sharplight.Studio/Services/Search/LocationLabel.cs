namespace Sharplight.Studio.Services.Search;

/// <summary>Names where in a document a passage is, in that format's words (page, slide, sheet, line).</summary>
internal static class LocationLabel
{
    /// <summary>For example <c>page 3</c>, <c>slide 2</c>, <c>line 41</c>, or empty when the passage has no position.</summary>
    /// <param name="file">The document (its extension tells what a page is).</param>
    /// <param name="page">The page, slide or sheet number, or 0.</param>
    /// <param name="line">The line, or 0.</param>
    /// <param name="heading">The heading the passage stands under, or empty.</param>
    public static string For(string file, int page, int line, string heading)
    {
        string where = "";
        if (page > 0)
        {
            string unit = Path.GetExtension(file).ToLowerInvariant() switch
            {
                ".pptx" => "slide",
                ".xlsx" => "sheet",
                _ => "page",
            };
            where = $"{unit} {page}";
        }
        else if (line > 0)
        {
            where = $"line {line}";
        }

        if (heading.Length == 0)
        {
            return where;
        }

        return where.Length == 0 ? heading : $"{where} · {heading}";
    }
}
