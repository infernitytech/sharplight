using System.Globalization;
using System.Text;

namespace Sharplight.Architectures;

/// <summary>
/// OmniVoice's rule-based length estimate (the official package's <c>RuleDurationEstimator</c>): every character weighs by its script (a Latin letter 1, a CJK ideograph 3, a digit 3.5,
/// punctuation 0.5, a space 0.2, a combining mark 0), and the target text takes as many audio frames as its weight is to the reference text's weight times the reference's frames; short
/// estimates are boosted by a power curve.
/// </summary>
internal static class OmniVoiceDurationEstimator
{
    private static readonly Dictionary<string, double> s_weights = new(StringComparer.Ordinal)
    {
        ["cjk"] = 3.0,
        ["hangul"] = 2.5,
        ["kana"] = 2.2,
        ["ethiopic"] = 3.0,
        ["yi"] = 3.0,
        ["indic"] = 1.8,
        ["thai_lao"] = 1.5,
        ["khmer_myanmar"] = 1.8,
        ["arabic"] = 1.5,
        ["hebrew"] = 1.5,
        ["latin"] = 1.0,
        ["cyrillic"] = 1.0,
        ["greek"] = 1.0,
        ["armenian"] = 1.0,
        ["georgian"] = 1.0,
        ["punctuation"] = 0.5,
        ["space"] = 0.2,
        ["digit"] = 3.5,
        ["mark"] = 0.0,
        ["default"] = 1.0,
    };

    // (last code point of the block, script): binary-searched for a character's script.
    private static readonly (int End, string Script)[] s_ranges =
    [
        (0x02AF, "latin"),
        (0x03FF, "greek"),
        (0x052F, "cyrillic"),
        (0x058F, "armenian"),
        (0x05FF, "hebrew"),
        (0x077F, "arabic"),
        (0x089F, "arabic"),
        (0x08FF, "arabic"),
        (0x097F, "indic"),
        (0x09FF, "indic"),
        (0x0A7F, "indic"),
        (0x0AFF, "indic"),
        (0x0B7F, "indic"),
        (0x0BFF, "indic"),
        (0x0C7F, "indic"),
        (0x0CFF, "indic"),
        (0x0D7F, "indic"),
        (0x0DFF, "indic"),
        (0x0EFF, "thai_lao"),
        (0x0FFF, "indic"),
        (0x109F, "khmer_myanmar"),
        (0x10FF, "georgian"),
        (0x11FF, "hangul"),
        (0x137F, "ethiopic"),
        (0x139F, "ethiopic"),
        (0x13FF, "default"),
        (0x167F, "default"),
        (0x169F, "default"),
        (0x16FF, "default"),
        (0x171F, "default"),
        (0x173F, "default"),
        (0x175F, "default"),
        (0x177F, "default"),
        (0x17FF, "khmer_myanmar"),
        (0x18AF, "default"),
        (0x18FF, "default"),
        (0x194F, "indic"),
        (0x19DF, "indic"),
        (0x19FF, "khmer_myanmar"),
        (0x1A1F, "indic"),
        (0x1AAF, "indic"),
        (0x1B7F, "indic"),
        (0x1BBF, "indic"),
        (0x1BFF, "indic"),
        (0x1C4F, "indic"),
        (0x1C7F, "indic"),
        (0x1C8F, "cyrillic"),
        (0x1CBF, "georgian"),
        (0x1CCF, "indic"),
        (0x1CFF, "indic"),
        (0x1D7F, "latin"),
        (0x1DBF, "latin"),
        (0x1DFF, "default"),
        (0x1EFF, "latin"),
        (0x309F, "kana"),
        (0x30FF, "kana"),
        (0x312F, "cjk"),
        (0x318F, "hangul"),
        (0x9FFF, "cjk"),
        (0xA4CF, "yi"),
        (0xA4FF, "default"),
        (0xA63F, "default"),
        (0xA69F, "cyrillic"),
        (0xA6FF, "default"),
        (0xA7FF, "latin"),
        (0xA82F, "indic"),
        (0xA87F, "default"),
        (0xA8DF, "indic"),
        (0xA8FF, "indic"),
        (0xA92F, "indic"),
        (0xA95F, "indic"),
        (0xA97F, "hangul"),
        (0xA9DF, "indic"),
        (0xA9FF, "khmer_myanmar"),
        (0xAA5F, "indic"),
        (0xAA7F, "khmer_myanmar"),
        (0xAADF, "indic"),
        (0xAAFF, "indic"),
        (0xAB2F, "ethiopic"),
        (0xAB6F, "latin"),
        (0xABBF, "default"),
        (0xABFF, "indic"),
        (0xD7AF, "hangul"),
        (0xFAFF, "cjk"),
        (0xFDFF, "arabic"),
        (0xFE6F, "default"),
        (0xFEFF, "arabic"),
        (0xFFEF, "latin"),
    ];

    private static readonly int[] s_breakpoints = [.. s_ranges.Select(r => r.End)];

    /// <summary>The weight of a character.</summary>
    /// <param name="rune">A Unicode scalar value.</param>
    public static double Weight(Rune rune)
    {
        int code = rune.Value;
        if ((code is >= 65 and <= 90) || (code is >= 97 and <= 122))
        {
            return s_weights["latin"];
        }

        if (code == 32)
        {
            return s_weights["space"];
        }

        if (code == 0x0640)
        {
            return s_weights["mark"];
        }

        var category = Rune.GetUnicodeCategory(rune);
        switch (category)
        {
            case UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark:
                return s_weights["mark"];
            case UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation
                or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation or UnicodeCategory.OtherPunctuation or UnicodeCategory.MathSymbol
                or UnicodeCategory.CurrencySymbol or UnicodeCategory.ModifierSymbol or UnicodeCategory.OtherSymbol:
                return s_weights["punctuation"];
            case UnicodeCategory.SpaceSeparator or UnicodeCategory.LineSeparator or UnicodeCategory.ParagraphSeparator:
                return s_weights["space"];
            case UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber:
                return s_weights["digit"];
        }

        int index = Array.BinarySearch(s_breakpoints, code);
        if (index < 0)
        {
            index = ~index;
        }

        if (index < s_ranges.Length)
        {
            return s_weights.GetValueOrDefault(s_ranges[index].Script, s_weights["default"]);
        }

        return code > 0x20000 ? s_weights["cjk"] : s_weights["default"];
    }

    /// <summary>The summed weight of a text.</summary>
    /// <param name="text">The text.</param>
    public static double TotalWeight(string text)
    {
        double total = 0;
        foreach (var rune in text.EnumerateRunes())
        {
            total += Weight(rune);
        }

        return total;
    }

    /// <summary>The estimated duration of <paramref name="target"/> from a reference of known text and length.</summary>
    /// <param name="target">The text to speak.</param>
    /// <param name="reference">The reference text.</param>
    /// <param name="referenceDuration">The reference's length (frames).</param>
    /// <param name="lowThreshold">Estimates below this are boosted.</param>
    /// <param name="boostStrength">How strongly short estimates are boosted (1 = linear).</param>
    public static double Estimate(string target, string reference, double referenceDuration, double lowThreshold = 50, double boostStrength = 3)
    {
        if (referenceDuration <= 0 || reference.Length == 0)
        {
            return 0;
        }

        double referenceWeight = TotalWeight(reference);
        if (referenceWeight == 0)
        {
            return 0;
        }

        double speed = referenceWeight / referenceDuration;
        double estimated = TotalWeight(target) / speed;
        return estimated < lowThreshold ? lowThreshold * Math.Pow(estimated / lowThreshold, 1.0 / boostStrength) : estimated;
    }
}
