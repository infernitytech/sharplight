namespace Sharplight.Studio.Services.Search;

/// <summary>Okapi BM25 keyword scoring over a list of passages (lower-cased letter and digit words, no stemming).</summary>
internal sealed class Bm25Index
{
    private const double K1 = 1.2;
    private const double B = 0.75;

    private readonly Dictionary<string, List<(int Document, int Frequency)>> _postings = new(StringComparer.Ordinal);
    private readonly int[] _lengths;
    private readonly double _averageLength;

    /// <param name="documents">The passages; a result's index is the passage's position in this list.</param>
    public Bm25Index(IReadOnlyList<string> documents)
    {
        _lengths = new int[documents.Count];
        long total = 0;
        for (int i = 0; i < documents.Count; i++)
        {
            var counts = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (string word in Tokenize(documents[i]))
            {
                counts[word] = counts.GetValueOrDefault(word) + 1;
                _lengths[i]++;
            }

            total += _lengths[i];
            foreach (var (word, count) in counts)
            {
                if (!_postings.TryGetValue(word, out var list))
                {
                    list = [];
                    _postings[word] = list;
                }

                list.Add((i, count));
            }
        }

        _averageLength = documents.Count == 0 ? 0 : (double)total / documents.Count;
    }

    /// <summary>How many passages are indexed.</summary>
    public int Count => _lengths.Length;

    /// <summary>The lower-cased words (runs of letters and digits) of <paramref name="text"/>.</summary>
    /// <param name="text">Any text.</param>
    public static IEnumerable<string> Tokenize(string text)
    {
        int start = -1;
        for (int i = 0; i <= text.Length; i++)
        {
            bool word = i < text.Length && char.IsLetterOrDigit(text[i]);
            if (word && start < 0)
            {
                start = i;
            }
            else if (!word && start >= 0)
            {
                yield return text[start..i].ToLowerInvariant();
                start = -1;
            }
        }
    }

    /// <summary>The <paramref name="count"/> best passages for <paramref name="query"/>, best first; passages with no query word are left out.</summary>
    /// <param name="query">The question.</param>
    /// <param name="count">How many passages.</param>
    public IReadOnlyList<(int Index, double Score)> Search(string query, int count)
    {
        var scores = new Dictionary<int, double>();
        foreach (string word in Tokenize(query).Distinct(StringComparer.Ordinal))
        {
            if (!_postings.TryGetValue(word, out var list))
            {
                continue;
            }

            double idf = Math.Log(1 + (Count - list.Count + 0.5) / (list.Count + 0.5));
            foreach (var (document, frequency) in list)
            {
                double normalization = 1 - B + B * _lengths[document] / Math.Max(1e-9, _averageLength);
                scores[document] = scores.GetValueOrDefault(document) + idf * frequency * (K1 + 1) / (frequency + K1 * normalization);
            }
        }

        return [.. scores.OrderByDescending(pair => pair.Value).ThenBy(pair => pair.Key).Take(count).Select(pair => (pair.Key, pair.Value))];
    }
}
