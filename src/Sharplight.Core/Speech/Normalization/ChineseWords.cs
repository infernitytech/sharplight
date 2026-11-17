namespace Sharplight.Speech.Normalization;

/// <summary>Chinese number words (<c>一百零一</c>, <c>一千二百三十四</c>, <c>一万</c>); years are read digit by digit.</summary>
internal sealed class ChineseWords : INumberWords
{
    private const string Digits = "零一二三四五六七八九";

    /// <inheritdoc/>
    public string Cardinal(long number)
    {
        if (number == 0)
        {
            return "零";
        }

        if (number >= 100_000_000)
        {
            long high = number / 100_000_000;
            long rest = number % 100_000_000;
            return Cardinal(high) + "亿" + (rest == 0 ? "" : rest < 10_000_000 ? "零" + Cardinal(rest) : Cardinal(rest));
        }

        if (number >= 10_000)
        {
            long high = number / 10_000;
            long rest = number % 10_000;
            return Cardinal(high) + "万" + (rest == 0 ? "" : rest < 1000 ? "零" + Cardinal(rest) : Cardinal(rest));
        }

        return Below10000((int)number);
    }

    /// <inheritdoc/>
    public string Ordinal(long number) => "第" + Cardinal(number);

    /// <inheritdoc/>
    public string Year(long year) => string.Concat(year.ToString(System.Globalization.CultureInfo.InvariantCulture).Select(c => Digits[c - '0']));

    /// <inheritdoc/>
    public string Digit(int digit) => Digits[digit].ToString();

    private static string Below10000(int n)
    {
        var text = new System.Text.StringBuilder();
        int[] units = [1000, 100, 10, 1];
        string[] names = ["千", "百", "十", ""];
        bool pendingZero = false;
        for (int i = 0; i < 4; i++)
        {
            int digit = n / units[i] % 10;
            if (digit == 0)
            {
                pendingZero = text.Length > 0;
                continue;
            }

            if (pendingZero)
            {
                text.Append('零');
                pendingZero = false;
            }

            // 10 to 19 are read 十, 十一, ... without the leading 一.
            if (!(digit == 1 && i == 2 && text.Length == 0))
            {
                text.Append(Digits[digit]);
            }

            text.Append(names[i]);
        }

        return text.ToString();
    }
}
