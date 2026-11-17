namespace Sharplight.Speech.Normalization;

/// <summary>Portuguese number words (Brazilian short scale: <c>cento e vinte e três</c>, <c>mil e cem</c>, <c>dois milhões</c>).</summary>
internal sealed class PortugueseWords : INumberWords
{
    private static readonly string[] s_ones =
    [
        "zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez", "onze", "doze", "treze", "catorze", "quinze", "dezesseis",
        "dezessete", "dezoito", "dezenove",
    ];

    private static readonly string[] s_tens = ["", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa"];

    private static readonly string[] s_hundreds =
        ["", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos", "seiscentos", "setecentos", "oitocentos", "novecentos"];

    private static readonly string[] s_ordinals = ["", "primeiro", "segundo", "terceiro", "quarto", "quinto", "sexto", "sétimo", "oitavo", "nono", "décimo"];

    /// <inheritdoc/>
    public string Cardinal(long number) => number == 0 ? "zero" : Compose(number);

    /// <inheritdoc/>
    public string Ordinal(long number) => number is >= 1 and <= 10 ? s_ordinals[number] : Cardinal(number);

    /// <inheritdoc/>
    public string Year(long year) => Cardinal(year);

    /// <inheritdoc/>
    public string Digit(int digit) => s_ones[digit];

    private static string Compose(long n)
    {
        if (n < 1000)
        {
            return Below1000((int)n);
        }

        long scale = n < 1_000_000 ? 1000 : n < 1_000_000_000 ? 1_000_000 : n < 1_000_000_000_000 ? 1_000_000_000 : 1_000_000_000_000;
        long high = n / scale;
        long rest = n % scale;
        string head = scale switch
        {
            1000 => high == 1 ? "mil" : Below1000((int)high) + " mil",
            1_000_000 => high == 1 ? "um milhão" : Compose(high) + " milhões",
            1_000_000_000 => high == 1 ? "um bilhão" : Compose(high) + " bilhões",
            _ => high == 1 ? "um trilhão" : Compose(high) + " trilhões",
        };
        if (rest == 0)
        {
            return head;
        }

        // "e" joins the last part when it is below one hundred or a round hundred (mil e cem, dois mil e vinte), else only a space (mil cento e um).
        string tail = Compose(rest);
        long last = rest;
        while (last % 1000 == 0)
        {
            last /= 1000;
        }

        bool joinWithAnd = last < 100 || last % 100 == 0 && last < 1000;
        return head + (joinWithAnd ? " e " : " ") + tail;
    }

    private static string Below1000(int n)
    {
        if (n == 100)
        {
            return "cem";
        }

        string text = n >= 100 ? s_hundreds[n / 100] : "";
        int rest = n % 100;
        if (rest == 0)
        {
            return text;
        }

        string tail = rest < 20 ? s_ones[rest] : s_tens[rest / 10] + (rest % 10 == 0 ? "" : " e " + s_ones[rest % 10]);
        return text.Length == 0 ? tail : text + " e " + tail;
    }
}
