using System.Globalization;
using System.Text.RegularExpressions;

namespace Sharplight.Vision;

/// <summary>
/// Reads the text of models that write locations as tokens (<c>&lt;loc_N&gt;</c>, N in [0, bins): the coordinate quantized over the image side): a label followed by four tokens is a
/// box (x1, y1, x2, y2) per group of four (a label may be followed by several boxes), or with <c>quads</c> a quadrilateral per group of eight (reported as its bounding box).
/// Text without location tokens has no regions.
/// </summary>
internal static partial class LocationTextParser
{
    [GeneratedRegex(@"([^<]*)((?:<loc_\d+>)+)")]
    private static partial Regex Located();

    [GeneratedRegex(@"<loc_(\d+)>")]
    private static partial Regex Location();

    [GeneratedRegex(@"</?s>|<pad>")]
    private static partial Regex Control();

    /// <summary>The text without the control tokens (<c>&lt;s&gt;</c>, <c>&lt;/s&gt;</c>, <c>&lt;pad&gt;</c>), trimmed.</summary>
    public static string Clean(string raw) => Control().Replace(raw, "").Trim();

    /// <summary>The located pieces of <paramref name="raw"/> (the decoded text with its special tokens) on an image of the given size, in pixels.</summary>
    /// <param name="raw">The decoded text.</param>
    /// <param name="bins">The number of coordinate bins.</param>
    /// <param name="width">The image width.</param>
    /// <param name="height">The image height.</param>
    /// <param name="quads">Whether a location is a quadrilateral (eight tokens) rather than a box (four).</param>
    public static IReadOnlyList<TextRegion> Parse(string raw, int bins, int width, int height, bool quads)
    {
        var regions = new List<TextRegion>();
        int group = quads ? 8 : 4;
        foreach (Match match in Located().Matches(Clean(raw)))
        {
            var values = Location().Matches(match.Groups[2].Value).Select(m => int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture)).ToArray();
            for (int start = 0; start + group <= values.Length; start += group)
            {
                var xs = Enumerable.Range(0, group / 2).Select(i => (values[start + (2 * i)] + 0.5f) / bins * width).ToArray();
                var ys = Enumerable.Range(0, group / 2).Select(i => (values[start + (2 * i) + 1] + 0.5f) / bins * height).ToArray();
                regions.Add(new TextRegion(match.Groups[1].Value.Trim(), new BoundingBox(xs.Min(), ys.Min(), xs.Max(), ys.Max())));
            }
        }

        return regions;
    }

    /// <summary>Whether the text holds location tokens.</summary>
    public static bool HasLocations(string raw) => raw.Contains("<loc_", StringComparison.Ordinal);
}
