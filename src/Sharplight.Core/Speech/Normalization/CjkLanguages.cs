namespace Sharplight.Speech.Normalization;

/// <summary>The data of Chinese and Japanese.</summary>
internal static class CjkLanguages
{
    /// <summary>Mandarin Chinese.</summary>
    public static CjkLanguage Chinese { get; } = new()
    {
        Code = "zh",
        Words = new ChineseWords(),
        YearByDigits = true,
        PercentPrefix = "百分之",
        PercentSuffix = "",
        Minus = "负",
        DecimalWord = "点",
        HourUnit = "点",
        MinuteUnit = "分",
        Currencies = new Dictionary<char, string> { ['¥'] = "元", ['￥'] = "元", ['$'] = "美元", ['€'] = "欧元", ['£'] = "英镑" },
        Symbols = [("+", "加"), ("=", "等于"), ("&", "和"), ("@", "艾特")],
        TwoBeforeMeasures = true,
    };

    /// <summary>Japanese.</summary>
    public static CjkLanguage Japanese { get; } = new()
    {
        Code = "ja",
        Words = new JapaneseWords(),
        YearByDigits = false,
        PercentPrefix = "",
        PercentSuffix = "パーセント",
        Minus = "マイナス",
        DecimalWord = "点",
        HourUnit = "時",
        MinuteUnit = "分",
        Currencies = new Dictionary<char, string> { ['¥'] = "円", ['￥'] = "円", ['$'] = "ドル", ['€'] = "ユーロ", ['£'] = "ポンド" },
        Symbols = [("+", "プラス"), ("=", "イコール"), ("&", "アンド"), ("@", "アット")],
    };
}
