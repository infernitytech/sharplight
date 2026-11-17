namespace Sharplight.Speech.Normalization;

/// <summary>Spanish number words (<c>veintitrés</c>, <c>treinta y uno</c>, <c>doscientos</c>, <c>un millón</c>).</summary>
internal sealed class SpanishWords : INumberWords
{
    private static readonly string[] s_ones =
    [
        "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis",
        "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete",
        "veintiocho", "veintinueve",
    ];

    private static readonly string[] s_tens = ["", "", "", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"];

    private static readonly string[] s_hundreds =
        ["", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos", "seiscientos", "setecientos", "ochocientos", "novecientos"];

    private static readonly string[] s_ordinals = ["", "primero", "segundo", "tercero", "cuarto", "quinto", "sexto", "séptimo", "octavo", "noveno", "décimo"];

    /// <inheritdoc/>
    public string Cardinal(long number) => Cardinal(number, apocopate: false);

    /// <inheritdoc/>
    public string Ordinal(long number) => number is >= 1 and <= 10 ? s_ordinals[number] : Cardinal(number);

    /// <inheritdoc/>
    public string Year(long year) => Cardinal(year);

    /// <inheritdoc/>
    public string Digit(int digit) => s_ones[digit];

    // Millions are "millones", a thousand millions "mil millones", the next scale (10^12) "billones"; "un" instead of "uno" before a scale word.
    private static string Cardinal(long n, bool apocopate)
    {
        if (n < 1000)
        {
            return n == 0 ? "cero" : Below1000((int)n, apocopate);
        }

        long scale = n < 1_000_000 ? 1000 : n < 1_000_000_000_000 ? 1_000_000 : 1_000_000_000_000;
        long high = n / scale;
        long rest = n % scale;
        string head = scale switch
        {
            1000 => high == 1 ? "mil" : Below1000((int)high, apocopate: true) + " mil",
            1_000_000 => high == 1 ? "un millón" : Cardinal(high, apocopate: true) + " millones",
            _ => high == 1 ? "un billón" : Cardinal(high, apocopate: true) + " billones",
        };
        return rest == 0 ? head : head + " " + Cardinal(rest, apocopate);
    }

    // "un" instead of "uno" (and "veintiún") before a noun or scale word.
    private static string Below1000(int n, bool apocopate)
    {
        if (n == 100)
        {
            return "cien";
        }

        string text = n >= 100 ? s_hundreds[n / 100] : "";
        int rest = n % 100;
        if (rest == 0)
        {
            return text;
        }

        string tail;
        if (rest < 30)
        {
            tail = s_ones[rest];
            if (apocopate && rest == 1)
            {
                tail = "un";
            }
            else if (apocopate && rest == 21)
            {
                tail = "veintiún";
            }
        }
        else
        {
            int unit = rest % 10;
            tail = s_tens[rest / 10] + (unit == 0 ? "" : " y " + (apocopate && unit == 1 ? "un" : s_ones[unit]));
        }

        return text.Length == 0 ? tail : text + " " + tail;
    }
}
