using System.Text.RegularExpressions;

namespace Sharplight.Onnx.Conformance;

/// <summary>
/// The reviewed list of node tests that are expected not to pass (conformance-skips.txt). Each line is
/// <c>&lt;provider&gt; &lt;test name or glob&gt; &lt;reason&gt;</c> where provider is <c>cpu</c>, <c>vulkan</c> or <c>*</c>; <c>#</c> starts a comment.
/// </summary>
internal static class SkipList
{
    private static readonly Lazy<List<(string Provider, Regex Pattern, string Reason)>> s_entries = new(Load);

    /// <summary>The skip reason of a test on a provider, or null when the test must pass.</summary>
    public static string? Reason(string provider, string test)
    {
        foreach (var (p, pattern, reason) in s_entries.Value)
        {
            if ((p == "*" || p == provider) && pattern.IsMatch(test))
            {
                return reason;
            }
        }

        return null;
    }

    private static List<(string, Regex, string)> Load()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "conformance-skips.txt");
        var entries = new List<(string, Regex, string)>();
        foreach (var raw in File.Exists(path) ? File.ReadAllLines(path) : [])
        {
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
            {
                continue;
            }

            var parts = line.Split((char[]?)null, 3, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 3)
            {
                throw new InvalidDataException($"conformance-skips.txt: '{line}' needs '<provider> <test or glob> <reason>'");
            }

            var regex = new Regex("^" + Regex.Escape(parts[1]).Replace("\\*", ".*").Replace("\\?", ".") + "$", RegexOptions.CultureInvariant);
            entries.Add((parts[0], regex, parts[2]));
        }

        return entries;
    }
}
