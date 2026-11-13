using System.Globalization;

namespace Sharplight.Tokenizers.Bpe;

/// <summary>
/// Code point classification used by the pre-tokenizers. Mirrors llama.cpp's <c>unicode_cpt_flags</c>:
/// letter = \p{L}, number = \p{N}, mark = \p{M}, whitespace = llama.cpp's explicit whitespace set.
/// Every in-range code point carries <see cref="Present"/>; out-of-range lookups return 0.
/// </summary>
internal static class UnicodeInfo
{
    public const byte Letter = 1;
    public const byte Number = 2;
    public const byte Mark = 4;
    public const byte Whitespace = 8;
    public const byte Present = 128;

    private static readonly byte[] Bmp = BuildBmp();

    private static byte[] BuildBmp()
    {
        var t = new byte[0x10000];
        for (int cp = 0; cp < t.Length; cp++)
        {
            t[cp] = Compute(cp);
        }

        return t;
    }

    public static byte Flags(int cp) => (uint)cp < 0x10000u ? Bmp[cp] : Compute(cp);

    private static byte Compute(int cp)
    {
        byte f = Present;
        if (IsWhitespace(cp))
        {
            f |= Whitespace;
        }

        if (cp >= 0xD800 && cp <= 0xDFFF)
        {
            return f;
        }

        if (cp > 0x10FFFF)
        {
            return f;
        }

        switch (CharUnicodeInfo.GetUnicodeCategory(cp))
        {
            case UnicodeCategory.UppercaseLetter:
            case UnicodeCategory.LowercaseLetter:
            case UnicodeCategory.TitlecaseLetter:
            case UnicodeCategory.ModifierLetter:
            case UnicodeCategory.OtherLetter:
                f |= Letter;
                break;
            case UnicodeCategory.DecimalDigitNumber:
            case UnicodeCategory.LetterNumber:
            case UnicodeCategory.OtherNumber:
                f |= Number;
                break;
            case UnicodeCategory.NonSpacingMark:
            case UnicodeCategory.SpacingCombiningMark:
            case UnicodeCategory.EnclosingMark:
                f |= Mark;
                break;
        }
        return f;
    }

    /// <summary>llama.cpp's <c>unicode_set_whitespace</c>.</summary>
    public static bool IsWhitespace(int cp) => cp switch
    {
        >= 0x09 and <= 0x0D => true,
        0x20 or 0x85 or 0xA0 or 0x1680 => true,
        >= 0x2000 and <= 0x200A => true,
        0x2028 or 0x2029 or 0x202F or 0x205F or 0x3000 => true,
        _ => false,
    };
}
