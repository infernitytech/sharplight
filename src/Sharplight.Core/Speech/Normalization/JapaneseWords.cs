namespace Sharplight.Speech.Normalization;

/// <summary>Japanese number words in kanji (<c>百二十三</c>, <c>千</c>, <c>一万</c>, <c>二千二十四</c>).</summary>
internal sealed class JapaneseWords : INumberWords
{
    private const string Digits = "〇一二三四五六七八九";

    /// <inheritdoc/>
    public string Cardinal(long number)
    {
        if (number == 0)
        {
            return "零";
        }

        string[] bigNames = ["", "万", "億", "兆"];
        var parts = new List<string>();
        int index = 0;
        while (number > 0)
        {
            int group = (int)(number % 10_000);
            if (group > 0)
            {
                // 一 stays before 万, 億 and 兆 (一万) but not before 千 (千).
                parts.Insert(0, Below10000(group, keepOne: index > 0) + bigNames[index]);
            }

            number /= 10_000;
            index++;
        }

        return string.Concat(parts);
    }

    /// <inheritdoc/>
    public string Ordinal(long number) => Cardinal(number) + "番目";

    /// <inheritdoc/>
    public string Year(long year) => Cardinal(year);

    /// <inheritdoc/>
    public string Digit(int digit) => Digits[digit].ToString();

    private static string Below10000(int n, bool keepOne)
    {
        if (n == 1 && keepOne)
        {
            return "一";
        }

        var text = new System.Text.StringBuilder();
        int[] units = [1000, 100, 10, 1];
        string[] names = ["千", "百", "十", ""];
        for (int i = 0; i < 4; i++)
        {
            int digit = n / units[i] % 10;
            if (digit == 0)
            {
                continue;
            }

            if (digit != 1 || i == 3)
            {
                text.Append(Digits[digit]);
            }

            text.Append(names[i]);
        }

        return text.ToString();
    }
}
