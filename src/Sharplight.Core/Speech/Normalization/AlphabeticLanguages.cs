namespace Sharplight.Speech.Normalization;

/// <summary>The data of English, German, French, Spanish, Italian and Portuguese.</summary>
internal static class AlphabeticLanguages
{
    /// <summary>English.</summary>
    public static AlphabeticLanguage English { get; } = new()
    {
        CurrencyJoiner = "and",
        Code = "en",
        Words = new EnglishWords(),
        DecimalSeparator = '.',
        ThousandsSeparators = ",",
        DecimalWord = "point",
        Minus = "minus",
        Percent = "percent",
        Months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"],
        Currencies = Currencies(("$", "dollar", "dollars", "cent", "cents"), ("USD", "dollar", "dollars", "cent", "cents"), ("€", "euro", "euros", "cent", "cents"),
            ("EUR", "euro", "euros", "cent", "cents"), ("£", "pound", "pounds", "penny", "pence"), ("GBP", "pound", "pounds", "penny", "pence")),
        Abbreviations =
        [
            ("Mr.", "Mister"), ("Mrs.", "Missus"), ("Ms.", "Miss"), ("Dr.", "Doctor"), ("Prof.", "Professor"), ("St.", "Saint"), ("Jr.", "Junior"), ("Sr.", "Senior"),
            ("vs.", "versus"), ("etc.", "et cetera"), ("e.g.", "for example"), ("i.e.", "that is"), ("approx.", "approximately"), ("Inc.", "Incorporated"),
            ("Ltd.", "Limited"), ("Co.", "Company"), ("No.", "number"), ("Gen.", "General"), ("Col.", "Colonel"), ("Capt.", "Captain"), ("Lt.", "Lieutenant"),
            ("Sgt.", "Sergeant"), ("Rev.", "Reverend"), ("Mt.", "Mount"), ("Ave.", "Avenue"), ("Blvd.", "Boulevard"), ("Rd.", "Road"),
        ],
        Symbols = [("&", "and"), ("+", "plus"), ("=", "equals"), ("@", "at")],
        MonthFirst = true,
        NumericMonthFirst = true,
        DayOrdinal = true,
        OrdinalPattern = @"(?<![\p{L}\d])(\d+)(?:st|nd|rd|th)(?![\p{L}\d])",
        TimeJoiner = "",
        HourSuffix = "o'clock",
    };

    /// <summary>German.</summary>
    public static AlphabeticLanguage German { get; } = new()
    {
        CurrencyJoiner = "und",
        Code = "de",
        Words = new GermanWords(),
        DecimalSeparator = ',',
        ThousandsSeparators = ". \u00A0\u202F",
        DecimalWord = "Komma",
        Minus = "minus",
        Percent = "Prozent",
        Months = ["januar", "februar", "märz", "april", "mai", "juni", "juli", "august", "september", "oktober", "november", "dezember"],
        Currencies = Currencies(("€", "Euro", "Euro", "Cent", "Cent"), ("EUR", "Euro", "Euro", "Cent", "Cent"), ("$", "Dollar", "Dollar", "Cent", "Cent"),
            ("USD", "Dollar", "Dollar", "Cent", "Cent"), ("£", "Pfund", "Pfund", "Penny", "Pence"), ("CHF", "Franken", "Franken", "Rappen", "Rappen")),
        Abbreviations =
        [
            ("z. B.", "zum Beispiel"), ("z.B.", "zum Beispiel"), ("d. h.", "das heißt"), ("d.h.", "das heißt"), ("u. a.", "unter anderem"), ("u.a.", "unter anderem"),
            ("usw.", "und so weiter"), ("bzw.", "beziehungsweise"), ("ca.", "circa"), ("Nr.", "Nummer"), ("Dr.", "Doktor"), ("Prof.", "Professor"), ("Hr.", "Herr"),
            ("Fr.", "Frau"), ("evtl.", "eventuell"), ("ggf.", "gegebenenfalls"), ("inkl.", "inklusive"), ("Str.", "Straße"), ("sog.", "sogenannte"), ("etc.", "et cetera"),
            ("vgl.", "vergleiche"), ("max.", "maximal"), ("min.", "mindestens"), ("Tel.", "Telefon"), ("Mio.", "Millionen"), ("Mrd.", "Milliarden"),
        ],
        Symbols = [("&", "und"), ("+", "plus"), ("=", "gleich"), ("@", "at"), ("§", "Paragraph")],
        DayOrdinal = true,
        DayOrdinalSuffix = "n",
        TimeJoiner = "Uhr",
        HourSuffix = "Uhr",
    };

    /// <summary>French.</summary>
    public static AlphabeticLanguage French { get; } = new()
    {
        CurrencyJoiner = "et",
        Code = "fr",
        Words = new FrenchWords(),
        DecimalSeparator = ',',
        ThousandsSeparators = " \u00A0\u202F",
        DecimalWord = "virgule",
        Minus = "moins",
        Percent = "pour cent",
        Months = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
        Currencies = Currencies(("€", "euro", "euros", "centime", "centimes"), ("EUR", "euro", "euros", "centime", "centimes"), ("$", "dollar", "dollars", "cent", "cents"),
            ("USD", "dollar", "dollars", "cent", "cents"), ("£", "livre sterling", "livres sterling", "penny", "pence"), ("CHF", "franc suisse", "francs suisses", "centime", "centimes")),
        Abbreviations =
        [
            ("M.", "Monsieur"), ("MM.", "Messieurs"), ("Mme", "Madame"), ("Mlle", "Mademoiselle"), ("Dr", "Docteur"), ("Pr", "Professeur"), ("etc.", "et cetera"),
            ("p. ex.", "par exemple"), ("c.-à-d.", "c'est-à-dire"), ("env.", "environ"), ("n°", "numéro"), ("St", "Saint"), ("Ste", "Sainte"), ("Cie", "compagnie"),
            ("av.", "avenue"), ("bd", "boulevard"), ("tel.", "téléphone"), ("min.", "minimum"), ("max.", "maximum"),
        ],
        Symbols = [("&", "et"), ("+", "plus"), ("=", "égale"), ("@", "arobase")],
        OrdinalPattern = @"(?<![\p{L}\d])(\d+)(?:er|re|ère|e|ème|eme|è|nd|nde)(?![\p{L}\d])",
        TimeJoiner = "heures",
        HourSuffix = "heures",
        FirstOfMonth = "premier",
    };

    /// <summary>Spanish.</summary>
    public static AlphabeticLanguage Spanish { get; } = new()
    {
        CurrencyJoiner = "con",
        Code = "es",
        Words = new SpanishWords(),
        DecimalSeparator = ',',
        ThousandsSeparators = ". \u00A0\u202F",
        DecimalWord = "coma",
        Minus = "menos",
        Percent = "por ciento",
        Months = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
        Currencies = Currencies(("€", "euro", "euros", "céntimo", "céntimos"), ("EUR", "euro", "euros", "céntimo", "céntimos"), ("$", "dólar", "dólares", "centavo", "centavos"),
            ("USD", "dólar", "dólares", "centavo", "centavos"), ("£", "libra", "libras", "penique", "peniques")),
        Abbreviations =
        [
            ("Sr.", "señor"), ("Sra.", "señora"), ("Srta.", "señorita"), ("Dr.", "doctor"), ("Dra.", "doctora"), ("Ud.", "usted"), ("Uds.", "ustedes"), ("etc.", "etcétera"),
            ("p. ej.", "por ejemplo"), ("núm.", "número"), ("Av.", "avenida"), ("aprox.", "aproximadamente"), ("EE. UU.", "Estados Unidos"), ("EE.UU.", "Estados Unidos"),
            ("Prof.", "profesor"), ("tel.", "teléfono"),
        ],
        Symbols = [("&", "y"), ("+", "más"), ("=", "igual a"), ("@", "arroba")],
        DateJoiner = "de",
        OrdinalPattern = @"(?<![\p{L}\d])(\d+)\.?[ºª°](?![\p{L}\d])",
        TimeJoiner = "y",
        HourSuffix = "en punto",
        FirstOfMonth = "primero",
    };

    /// <summary>Italian.</summary>
    public static AlphabeticLanguage Italian { get; } = new()
    {
        CurrencyJoiner = "e",
        Code = "it",
        Words = new ItalianWords(),
        DecimalSeparator = ',',
        ThousandsSeparators = ". \u00A0\u202F",
        DecimalWord = "virgola",
        Minus = "meno",
        Percent = "per cento",
        Months = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"],
        Currencies = Currencies(("€", "euro", "euro", "centesimo", "centesimi"), ("EUR", "euro", "euro", "centesimo", "centesimi"), ("$", "dollaro", "dollari", "centesimo", "centesimi"),
            ("USD", "dollaro", "dollari", "centesimo", "centesimi"), ("£", "sterlina", "sterline", "penny", "penny")),
        Abbreviations =
        [
            ("Sig.", "signor"), ("Sig.ra", "signora"), ("Dott.", "dottor"), ("Dott.ssa", "dottoressa"), ("Prof.", "professor"), ("ecc.", "eccetera"), ("es.", "esempio"),
            ("ca.", "circa"), ("n.", "numero"), ("tel.", "telefono"), ("sig.", "signor"), ("avv.", "avvocato"), ("ing.", "ingegner"),
        ],
        Symbols = [("&", "e"), ("+", "più"), ("=", "uguale"), ("@", "chiocciola")],
        OrdinalPattern = @"(?<![\p{L}\d])(\d+)[°ºª](?![\p{L}\d])",
        TimeJoiner = "e",
        HourSuffix = "in punto",
        FirstOfMonth = "primo",
    };

    /// <summary>Portuguese (Brazilian spelling).</summary>
    public static AlphabeticLanguage Portuguese { get; } = new()
    {
        CurrencyJoiner = "e",
        Code = "pt",
        Words = new PortugueseWords(),
        DecimalSeparator = ',',
        ThousandsSeparators = ". \u00A0\u202F",
        DecimalWord = "vírgula",
        Minus = "menos",
        Percent = "por cento",
        Months = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"],
        Currencies = Currencies(("R$", "real", "reais", "centavo", "centavos"), ("BRL", "real", "reais", "centavo", "centavos"), ("€", "euro", "euros", "cêntimo", "cêntimos"),
            ("EUR", "euro", "euros", "cêntimo", "cêntimos"), ("$", "dólar", "dólares", "centavo", "centavos"), ("USD", "dólar", "dólares", "centavo", "centavos"),
            ("£", "libra", "libras", "péni", "pence")),
        Abbreviations =
        [
            ("Sr.", "senhor"), ("Sra.", "senhora"), ("Srta.", "senhorita"), ("Dr.", "doutor"), ("Dra.", "doutora"), ("Prof.", "professor"), ("etc.", "et cetera"),
            ("ex.", "exemplo"), ("nº", "número"), ("aprox.", "aproximadamente"), ("Av.", "avenida"), ("tel.", "telefone"),
        ],
        Symbols = [("&", "e"), ("+", "mais"), ("=", "igual a"), ("@", "arroba")],
        DateJoiner = "de",
        OrdinalPattern = @"(?<![\p{L}\d])(\d+)\.?[ºª°](?![\p{L}\d])",
        TimeJoiner = "e",
        HourSuffix = "horas",
        FirstOfMonth = "primeiro",
    };

    private static Dictionary<string, CurrencyNames> Currencies(params (string Symbol, string One, string Many, string SubOne, string SubMany)[] entries)
        => entries.ToDictionary(e => e.Symbol, e => new CurrencyNames(e.One, e.Many, e.SubOne, e.SubMany), StringComparer.Ordinal);
}
