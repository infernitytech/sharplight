using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// The constants of DeepSeek-V4.1's Engram n-gram memory that the release does not store but its reference code (<c>inference/engram.py</c>)
/// derives at load time, as the canonical <c>engram.*</c> hyper-parameters (<c>EngramOptions</c>): the bucket count of every hash head (distinct
/// primes above <c>engram_vocab_size</c>, never reused across layers), their first rows in the tables of all layers concatenated in layer
/// order, the odd hash multipliers (numpy's <c>default_rng(10007 * layer)</c>, bounded against int64 overflow) and the compressed token map
/// (tokens whose decoded text normalizes alike, NFKC, NFD, accents stripped, lower case, whitespace runs folded, share one hashed id).
/// </summary>
internal static partial class DeepseekV41Engram
{
    private const char Sentinel = '';

    /// <summary>Adds the <c>engram.*</c> hyper-parameters of <paramref name="c"/> under <paramref name="a"/>; nothing when the model has no Engram layers.</summary>
    /// <exception cref="InvalidDataException">The config and the derived tables disagree.</exception>
    public static void AddHyperParameters(Dictionary<string, object> md, string a, HfConfig c)
    {
        int[] layers = c.Get("engram_layer_ids") is { ValueKind: JsonValueKind.Array } ids ? [.. ids.EnumerateArray().Select(e => e.GetInt32())] : [];
        if (layers.Length == 0 || string.IsNullOrEmpty(c.Directory))
        {
            return;   // no Engram layers, or a config checked without its files (the token map needs tokenizer.json)
        }

        int order = c.Int("engram_max_ngram_size");
        int heads = c.Int("engram_n_heads");
        var sizes = Primes(layers.Length, order, heads, c.Int("engram_vocab_size"));
        long[] rows = c.Get("engram_num_embeddings") is { ValueKind: JsonValueKind.Array } counts ? [.. counts.EnumerateArray().Select(e => e.GetInt64())] : [];
        int perLayer = (order - 1) * heads;
        for (int layer = 0; layer < layers.Length; layer++)
        {
            long sum = sizes.Skip(layer * perLayer).Take(perLayer).Sum();
            if (layer >= rows.Length || rows[layer] != sum)
            {
                throw new InvalidDataException($"deepseek_v41: engram layer {layers[layer]} has {sum} hash buckets, engram_num_embeddings says otherwise.");
            }
        }

        var offsets = new long[sizes.Length];
        for (int i = 1; i < sizes.Length; i++)
        {
            offsets[i] = offsets[i - 1] + sizes[i - 1];
        }

        var (map, hashed) = TokenMap(Path.Combine(c.Directory, "tokenizer.json"), c.Int("vocab_size"));
        if (c.GetInt("engram_compressed_vocab_size") is { } declared && declared != hashed)
        {
            throw new InvalidDataException($"deepseek_v41: the tokenizer folds onto {hashed} hashed ids, engram_compressed_vocab_size is {declared}.");
        }

        md[a + "engram.layers"] = layers;
        md[a + "engram.order"] = order;
        md[a + "engram.heads_per_order"] = heads;
        md[a + "engram.multipliers"] = Multipliers(layers, order, hashed);
        md[a + "engram.head_sizes"] = sizes;
        md[a + "engram.head_offsets"] = offsets;
        md[a + "engram.token_map"] = map;
        md[a + "engram.pad_token"] = (long)c.Int("engram_pad_token_id", "engram_pad_id");
    }

    /// <summary>EngramLayout.from_args: per layer, per n-gram order 2..N, per head, the next unused prime above the previous one.</summary>
    internal static long[] Primes(int layers, int order, int heads, long vocabulary)
    {
        var seen = new HashSet<long>();
        var result = new List<long>();
        for (int layer = 0; layer < layers; layer++)
        {
            for (int n = 0; n < order - 1; n++)
            {
                long current = vocabulary - 1;
                for (int h = 0; h < heads; h++)
                {
                    long candidate = current + 1;
                    while (!IsPrime(candidate) || seen.Contains(candidate))
                    {
                        candidate++;
                    }

                    current = candidate;
                    seen.Add(current);
                    result.Add(current);
                }
            }
        }

        return [.. result];
    }

    private static bool IsPrime(long n)
    {
        if (n < 2 || n % 2 == 0)
        {
            return n == 2;
        }

        for (long d = 3; d * d <= n; d += 2)
        {
            if (n % d == 0)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>compute_hash_multipliers: N odd multipliers per layer.</summary>
    internal static long[] Multipliers(int[] layers, int order, int hashedVocabulary)
    {
        long bound = Math.Max(1, long.MaxValue / hashedVocabulary / 2);
        var result = new long[layers.Length * order];
        for (int i = 0; i < layers.Length; i++)
        {
            var rng = new NumpyDefaultRng((ulong)(10007L * layers[i]));
            for (int j = 0; j < order; j++)
            {
                result[i * order + j] = rng.NextBelow(bound) * 2 + 1;
            }
        }

        return result;
    }

    /// <summary>build_compressed_token_map over a byte-level BPE <c>tokenizer.json</c>: hashed id of every token id, and the hashed vocabulary size.</summary>
    /// <exception cref="InvalidDataException">The tokenizer is not byte-level BPE.</exception>
    internal static (int[] Map, int Hashed) TokenMap(string tokenizerPath, int vocabulary)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(tokenizerPath));
        var root = document.RootElement;
        if (!root.TryGetProperty("decoder", out var decoder) || decoder.ValueKind != JsonValueKind.Object
            || decoder.GetProperty("type").GetString() != "ByteLevel")
        {
            throw new InvalidDataException($"{tokenizerPath}: the Engram token map needs a byte-level tokenizer.");
        }

        var texts = new string?[vocabulary];
        var raw = new string?[vocabulary];
        var bytes = ByteLevelBytes();
        foreach (var entry in root.GetProperty("model").GetProperty("vocab").EnumerateObject())
        {
            int id = entry.Value.GetInt32();
            if (id < vocabulary)
            {
                raw[id] = entry.Name;
                texts[id] = Encoding.UTF8.GetString([.. entry.Name.Select(ch => bytes.TryGetValue(ch, out var b) ? b : (byte)'?')]);
            }
        }

        foreach (var added in root.GetProperty("added_tokens").EnumerateArray())
        {
            int id = added.GetProperty("id").GetInt32();
            if (id < vocabulary)
            {
                raw[id] = texts[id] = added.GetProperty("content").GetString();
            }
        }

        var keys = new Dictionary<string, int>(StringComparer.Ordinal);
        var map = new int[vocabulary];
        for (int id = 0; id < vocabulary; id++)
        {
            string key;
            if (texts[id] is not { } text)
            {
                key = "\0" + id.ToString(CultureInfo.InvariantCulture);   // an id without a token keeps its own hashed id
            }
            else if (text.Contains('�', StringComparison.Ordinal))
            {
                key = raw[id]!;
            }
            else
            {
                string normalized = Normalize(text);
                key = normalized.Length > 0 ? normalized : text;
            }

            if (!keys.TryGetValue(key, out int hashed))
            {
                keys[key] = hashed = keys.Count;
            }

            map[id] = hashed;
        }

        return (map, keys.Count);
    }

    // The normalizer sequence of build_compressed_token_map (tokenizers: NFKC, NFD, StripAccents, Lowercase, whitespace folding, Strip).
    internal static string Normalize(string text)
    {
        string decomposed = text.Normalize(NormalizationForm.FormKC).Normalize(NormalizationForm.FormD);
        var stripped = new StringBuilder(decomposed.Length);
        foreach (var rune in decomposed.EnumerateRunes())
        {
            if (Rune.GetUnicodeCategory(rune) is not (UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark))
            {
                stripped.Append(rune.ToString());
            }
        }

        string folded = Whitespace().Replace(Lower(stripped.ToString()), " ");
        if (folded == " ")
        {
            folded = Sentinel.ToString();
        }

        return folded.Trim().Replace(Sentinel, ' ');
    }

    // Full Unicode lower-casing as Rust's str::to_lowercase: the dotted capital I becomes "i̇" and a final capital sigma "ς".
    private static string Lower(string text)
    {
        var result = new StringBuilder(text.Length);
        var runes = text.EnumerateRunes().ToArray();
        for (int i = 0; i < runes.Length; i++)
        {
            var rune = runes[i];
            if (rune.Value == 0x130)
            {
                result.Append("i̇");
            }
            else if (rune.Value == 0x3A3)
            {
                bool before = i > 0 && Rune.IsLetter(runes[i - 1]);
                bool after = i + 1 < runes.Length && Rune.IsLetter(runes[i + 1]);
                result.Append(before && !after ? 'ς' : 'σ');
            }
            else
            {
                result.Append(Rune.ToLowerInvariant(rune).ToString());
            }
        }

        return result.ToString();
    }

    [GeneratedRegex("[ \t\r\n]+")]
    private static partial Regex Whitespace();

    // GPT-2's bytes_to_unicode, inverted.
    private static Dictionary<char, byte> ByteLevelBytes()
    {
        var map = new Dictionary<char, byte>();
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            bool printable = b is >= '!' and <= '~' or >= 0xA1 and <= 0xAC or >= 0xAE and <= 0xFF;
            map[(char)(printable ? b : 256 + extra++)] = (byte)b;
        }

        return map;
    }
}
