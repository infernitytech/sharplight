using System.Text.RegularExpressions;

namespace Sharplight.Architectures;

/// <summary>
/// OmniVoice's voice-design instruction (the official package's <c>_resolve_instruct</c>): a comma-separated list of speaker attributes (gender, age, pitch, whisper, an English accent or a
/// Chinese dialect), validated, normalized to one language and checked for conflicts within a category.
/// </summary>
internal static partial class OmniVoiceInstruct
{
    private static readonly (string English, string Chinese)[][] s_pairs =
    [
        [("male", "男"), ("female", "女")],
        [("child", "儿童"), ("teenager", "少年"), ("young adult", "青年"), ("middle-aged", "中年"), ("elderly", "老年")],
        [("very low pitch", "极低音调"), ("low pitch", "低音调"), ("moderate pitch", "中音调"), ("high pitch", "高音调"), ("very high pitch", "极高音调")],
        [("whisper", "耳语")],
    ];

    private static readonly string[] s_accents =
    [
        "american accent", "british accent", "australian accent", "chinese accent", "canadian accent", "indian accent", "korean accent", "portuguese accent", "russian accent", "japanese accent",
    ];

    private static readonly string[] s_dialects = ["河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"];

    private static readonly Dictionary<string, string> s_toChinese = s_pairs.SelectMany(c => c).ToDictionary(p => p.English, p => p.Chinese, StringComparer.Ordinal);
    private static readonly Dictionary<string, string> s_toEnglish = s_pairs.SelectMany(c => c).ToDictionary(p => p.Chinese, p => p.English, StringComparer.Ordinal);

    private static readonly HashSet<string> s_valid = [.. s_toChinese.Keys, .. s_toEnglish.Keys, .. s_accents, .. s_dialects];

    [GeneratedRegex(@"\s*[,，]\s*")]
    private static partial Regex Separator();

    /// <summary>The valid English items.</summary>
    public static IReadOnlyList<string> EnglishItems { get; } = [.. s_valid.Where(i => !HasIdeograph(i)).Order(StringComparer.Ordinal)];

    /// <summary>Validates and normalizes an instruction.</summary>
    /// <param name="instruct">The instruction, or <see langword="null"/>.</param>
    /// <param name="chinese">Whether to speak Chinese attributes by default (the text has Chinese in it); an accent forces English, a dialect Chinese.</param>
    /// <returns>The normalized instruction, or <see langword="null"/> for none.</returns>
    /// <exception cref="ArgumentException">An item is not supported, or two items of one category conflict.</exception>
    public static string? Resolve(string? instruct, bool chinese)
    {
        if (string.IsNullOrWhiteSpace(instruct))
        {
            return null;
        }

        var items = Separator().Split(instruct.Trim()).Where(i => i.Length > 0).Select(i => i.Trim().ToLowerInvariant()).ToList();
        var unknown = items.Where(i => !s_valid.Contains(i)).ToList();
        if (unknown.Count > 0)
        {
            throw new ArgumentException($"Unsupported voice instruction items: {string.Join(", ", unknown)}. Valid items: {string.Join(", ", EnglishItems)} (and the Chinese ones).", nameof(instruct));
        }

        bool dialect = items.Any(i => i.EndsWith('话'));
        bool accent = items.Any(i => i.Contains(" accent", StringComparison.Ordinal));
        if (dialect && accent)
        {
            throw new ArgumentException("A Chinese dialect and an English accent cannot be mixed in one instruction.", nameof(instruct));
        }

        if (dialect)
        {
            chinese = true;
        }
        else if (accent)
        {
            chinese = false;
        }

        items = [.. items.Select(i => chinese ? s_toChinese.GetValueOrDefault(i, i) : s_toEnglish.GetValueOrDefault(i, i))];
        var categories = s_pairs.Select(c => c.SelectMany(p => new[] { p.English, p.Chinese }).ToHashSet(StringComparer.Ordinal)).Append(s_accents.ToHashSet(StringComparer.Ordinal))
            .Append(s_dialects.ToHashSet(StringComparer.Ordinal));
        foreach (var category in categories)
        {
            var hits = items.Where(category.Contains).ToList();
            if (hits.Count > 1)
            {
                throw new ArgumentException($"Conflicting voice instruction items in one category: {string.Join(" vs ", hits)}.", nameof(instruct));
            }
        }

        return string.Join(items.Any(HasIdeograph) ? "，" : ", ", items);
    }

    /// <summary>Whether the text contains a CJK ideograph.</summary>
    /// <param name="text">The text.</param>
    public static bool HasIdeograph(string text) => text.Any(c => c is >= '一' and <= '鿿');
}
