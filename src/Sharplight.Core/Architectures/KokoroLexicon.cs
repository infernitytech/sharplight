using System.Text.Json;

namespace Sharplight.Architectures;

/// <summary>
/// A pronunciation dictionary in the format of the misaki English lexicons (<c>us_gold.json</c>, <c>us_silver.json</c>, ...): word to phonemes, where a
/// word with several pronunciations maps to an object by part of speech whose <c>DEFAULT</c> entry is used here. Lookups also find the capitalized and
/// lower-case form of a word.
/// </summary>
internal sealed class KokoroLexicon
{
    private readonly Dictionary<string, string> _words = new(StringComparer.Ordinal);

    /// <summary>The number of entries.</summary>
    public int Count => _words.Count;

    /// <summary>Adds the entries of a lexicon file; entries already present are kept (earlier files take precedence).</summary>
    /// <param name="json">The JSON bytes of the file.</param>
    public void Add(ReadOnlySpan<byte> json)
    {
        var reader = new Utf8JsonReader(json);
        using var document = JsonDocument.ParseValue(ref reader);
        foreach (var entry in document.RootElement.EnumerateObject())
        {
            string? phonemes = entry.Value.ValueKind switch
            {
                JsonValueKind.String => entry.Value.GetString(),
                JsonValueKind.Object => entry.Value.TryGetProperty("DEFAULT", out var d) && d.ValueKind == JsonValueKind.String ? d.GetString() : null,
                _ => null,
            };
            if (phonemes is not null)
            {
                _words.TryAdd(entry.Name, phonemes);
            }
        }
    }

    /// <summary>The phonemes of <paramref name="word"/> (exact, then lower-case, then capitalized), or <see langword="null"/>.</summary>
    /// <param name="word">The word.</param>
    public string? Find(string word)
    {
        if (_words.TryGetValue(word, out var exact))
        {
            return exact;
        }

        string lower = word.ToLowerInvariant();
        if (lower != word && _words.TryGetValue(lower, out var l))
        {
            return l;
        }

        string capital = lower.Length > 1 ? char.ToUpperInvariant(lower[0]) + lower[1..] : lower.ToUpperInvariant();
        return _words.TryGetValue(capital, out var c) ? c : null;
    }
}
