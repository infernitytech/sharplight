using System.Text;
using System.Text.RegularExpressions;

namespace Sharplight.CatalogTool;

/// <summary>Matches repository paths against glob patterns: <c>*</c> within a path segment, <c>**</c> across segments, <c>?</c> one character.</summary>
internal static class GlobPattern
{
    /// <summary>Whether <paramref name="path"/> matches <paramref name="pattern"/> (case-sensitive, whole path).</summary>
    public static bool IsMatch(string pattern, string path) => ToRegex(pattern).IsMatch(path);

    private static Regex ToRegex(string pattern)
    {
        var text = new StringBuilder("^");
        for (int i = 0; i < pattern.Length; i++)
        {
            char c = pattern[i];
            if (c == '*' && i + 1 < pattern.Length && pattern[i + 1] == '*')
            {
                bool slash = i + 2 < pattern.Length && pattern[i + 2] == '/';
                text.Append(slash ? "(?:.*/)?" : ".*");
                i += slash ? 2 : 1;
            }
            else if (c == '*')
            {
                text.Append("[^/]*");
            }
            else if (c == '?')
            {
                text.Append("[^/]");
            }
            else
            {
                text.Append(Regex.Escape(c.ToString()));
            }
        }

        return new Regex(text.Append('$').ToString(), RegexOptions.CultureInvariant);
    }
}
