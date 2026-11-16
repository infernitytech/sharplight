namespace Sharplight.Speech;

/// <summary>
/// English names of spoken languages by ISO 639 code (the 99 languages of Whisper plus Filipino), for models that name languages in their
/// prompts or outputs ("language English") instead of using codes. Independent of the globalization data of the runtime.
/// </summary>
public static class SpeechLanguages
{
    private static readonly Dictionary<string, string> s_names = new(StringComparer.OrdinalIgnoreCase)
    {
        ["af"] = "Afrikaans", ["am"] = "Amharic", ["ar"] = "Arabic", ["as"] = "Assamese", ["az"] = "Azerbaijani", ["ba"] = "Bashkir",
        ["be"] = "Belarusian", ["bg"] = "Bulgarian", ["bn"] = "Bengali", ["bo"] = "Tibetan", ["br"] = "Breton", ["bs"] = "Bosnian",
        ["ca"] = "Catalan", ["cs"] = "Czech", ["cy"] = "Welsh", ["da"] = "Danish", ["de"] = "German", ["el"] = "Greek", ["en"] = "English",
        ["es"] = "Spanish", ["et"] = "Estonian", ["eu"] = "Basque", ["fa"] = "Persian", ["fi"] = "Finnish", ["fil"] = "Filipino",
        ["fo"] = "Faroese", ["fr"] = "French", ["gl"] = "Galician", ["gu"] = "Gujarati", ["ha"] = "Hausa", ["haw"] = "Hawaiian",
        ["he"] = "Hebrew", ["hi"] = "Hindi", ["hr"] = "Croatian", ["ht"] = "Haitian Creole", ["hu"] = "Hungarian", ["hy"] = "Armenian",
        ["id"] = "Indonesian", ["is"] = "Icelandic", ["it"] = "Italian", ["ja"] = "Japanese", ["jw"] = "Javanese", ["ka"] = "Georgian",
        ["kk"] = "Kazakh", ["km"] = "Khmer", ["kn"] = "Kannada", ["ko"] = "Korean", ["la"] = "Latin", ["lb"] = "Luxembourgish",
        ["ln"] = "Lingala", ["lo"] = "Lao", ["lt"] = "Lithuanian", ["lv"] = "Latvian", ["mg"] = "Malagasy", ["mi"] = "Maori",
        ["mk"] = "Macedonian", ["ml"] = "Malayalam", ["mn"] = "Mongolian", ["mr"] = "Marathi", ["ms"] = "Malay", ["mt"] = "Maltese",
        ["my"] = "Burmese", ["ne"] = "Nepali", ["nl"] = "Dutch", ["nn"] = "Nynorsk", ["no"] = "Norwegian", ["oc"] = "Occitan",
        ["pa"] = "Punjabi", ["pl"] = "Polish", ["ps"] = "Pashto", ["pt"] = "Portuguese", ["ro"] = "Romanian", ["ru"] = "Russian",
        ["sa"] = "Sanskrit", ["sd"] = "Sindhi", ["si"] = "Sinhala", ["sk"] = "Slovak", ["sl"] = "Slovenian", ["sn"] = "Shona",
        ["so"] = "Somali", ["sq"] = "Albanian", ["sr"] = "Serbian", ["su"] = "Sundanese", ["sv"] = "Swedish", ["sw"] = "Swahili",
        ["ta"] = "Tamil", ["te"] = "Telugu", ["tg"] = "Tajik", ["th"] = "Thai", ["tk"] = "Turkmen", ["tl"] = "Tagalog", ["tr"] = "Turkish",
        ["tt"] = "Tatar", ["uk"] = "Ukrainian", ["ur"] = "Urdu", ["uz"] = "Uzbek", ["vi"] = "Vietnamese", ["yi"] = "Yiddish",
        ["yo"] = "Yoruba", ["yue"] = "Cantonese", ["zh"] = "Chinese",
    };

    /// <summary>The English name of a language code, or the text itself when it is not a known code (e.g. already a name).</summary>
    /// <param name="language">An ISO 639 code such as <c>en</c>, or a name.</param>
    public static string EnglishName(string language)
    {
        ArgumentNullException.ThrowIfNull(language);
        return s_names.TryGetValue(language, out var name) ? name : language;
    }

    /// <summary>The ISO 639 code of an English language name, or the name itself when it is not known.</summary>
    /// <param name="name">A language name such as <c>English</c>.</param>
    public static string Code(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        foreach (var (code, english) in s_names)
        {
            if (string.Equals(english, name, StringComparison.OrdinalIgnoreCase))
            {
                return code;
            }
        }

        return name;
    }
}
