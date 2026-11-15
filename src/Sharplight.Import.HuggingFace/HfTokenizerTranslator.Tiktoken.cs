using System.Globalization;
using System.Text;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

// tiktoken vocabularies without tokenizer.json (Kimi K2 / Kimi Linear / Kimi K3: tiktoken.model + tokenization_kimi.py).
internal static partial class HfTokenizerTranslator
{
    /// <summary>The tiktoken rank file Kimi checkpoints ship.</summary>
    private const string TiktokenFile = "tiktoken.model";

    /// <summary>Kimi's remote tokenizer reserves this many special-token ids after the mergeable ranks.</summary>
    private const int KimiReservedSpecialTokens = 256;

    /// <summary>Whether <paramref name="dir"/> holds a Kimi tiktoken tokenizer (rank file plus Kimi's remote tokenizer code).</summary>
    private static bool IsKimiTiktoken(string dir)
        => File.Exists(Path.Combine(dir, TiktokenFile)) && File.Exists(Path.Combine(dir, "tokenization_kimi.py"));

    /// <summary>
    /// Byte-level BPE metadata of a Kimi tiktoken tokenizer, following llama.cpp's Kimi conversion: tokens are the mergeable ranks in
    /// GPT-2 byte-to-unicode spelling, merges are recovered from the ranks (each token split by rank-limited BPE), and the
    /// <see cref="KimiReservedSpecialTokens"/> ids after the ranks are control tokens named by <c>added_tokens_decoder</c>
    /// (<c>&lt;|reserved_token_{id}|&gt;</c> otherwise). The pre-tokenizer is <c>kimi-k2</c>.
    /// </summary>
    private static void AddKimiTiktoken(IDictionary<string, object> md, string dir, HfConfig config)
    {
        var ranks = ReadTiktokenRanks(Path.Combine(dir, TiktokenFile));
        var tokenizerConfig = ReadTokenizerConfig(dir);
        var decoderNames = new Dictionary<int, string>();
        if (tokenizerConfig is { ValueKind: JsonValueKind.Object } root
            && root.TryGetProperty("added_tokens_decoder", out var decoder) && decoder.ValueKind == JsonValueKind.Object)
        {
            foreach (var entry in decoder.EnumerateObject())
            {
                if (int.TryParse(entry.Name, NumberStyles.None, CultureInfo.InvariantCulture, out int id)
                    && entry.Value.TryGetProperty("content", out var content) && content.GetString() is { } text)
                {
                    decoderNames[id] = text;
                }
            }
        }

        int baseCount = ranks.Count;
        int vocabSize = config.GetInt("vocab_size") ?? baseCount + KimiReservedSpecialTokens;
        var byteChars = ByteToUnicode();
        var tokens = new string[vocabSize];
        var types = new int[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            tokens[i] = $"[PAD{i}]";
            types[i] = TypeUnused;
        }

        var spellings = new Dictionary<string, string>(ranks.Count, StringComparer.Ordinal);
        foreach (var (bytes, rank) in ranks)
        {
            string spelling = Spell(bytes, byteChars);
            spellings[bytes] = spelling;
            if (rank < vocabSize)
            {
                tokens[rank] = spelling;
                types[rank] = TypeNormal;
            }
        }

        var specials = new List<JsonElement>();
        for (int id = baseCount; id < Math.Min(baseCount + KimiReservedSpecialTokens, vocabSize); id++)
        {
            string name = decoderNames.TryGetValue(id, out var text) ? text : $"<|reserved_token_{id}|>";
            tokens[id] = name;
            types[id] = TypeControl;
            specials.Add(JsonSerializer.SerializeToElement(new Dictionary<string, object> { ["id"] = id, ["content"] = name, ["special"] = true }));
        }

        md[CanonicalKeys.TokenizerModel] = "gpt2";
        md[CanonicalKeys.TokenizerPre] = "kimi-k2";
        md[CanonicalKeys.TokenizerTokens] = tokens;
        md[CanonicalKeys.TokenizerTokenTypes] = types;
        md[CanonicalKeys.TokenizerMerges] = TiktokenMerges(ranks, spellings);

        using var empty = JsonDocument.Parse("{}");
        HfSpecialVocab.AddTo(md, dir, config, empty.RootElement, tokenizerConfig, [.. specials]);
    }

    /// <summary>The ranks of a tiktoken file (<c>base64 rank</c> per line), keyed by the token bytes as a Latin-1 string.</summary>
    private static Dictionary<string, int> ReadTiktokenRanks(string path)
    {
        var ranks = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var line in File.ReadLines(path))
        {
            if (line.Length == 0)
            {
                continue;
            }

            int space = line.IndexOf(' ');
            if (space <= 0)
            {
                throw new InvalidDataException($"{path}: malformed tiktoken line '{line}'.");
            }

            byte[] bytes = Convert.FromBase64String(line[..space]);
            ranks[Encoding.Latin1.GetString(bytes)] = int.Parse(line.AsSpan(space + 1), NumberStyles.None, CultureInfo.InvariantCulture);
        }

        return ranks;
    }

    /// <summary>
    /// The merges of a tiktoken vocabulary in rank order: each multi-byte token is split by byte-pair encoding with only the merges ranked
    /// below it; when that leaves exactly two parts, they are the token's merge (llama.cpp's <c>QwenModel.bpe</c>).
    /// </summary>
    private static string[] TiktokenMerges(Dictionary<string, int> ranks, Dictionary<string, string> spellings)
    {
        var merges = new List<string>(ranks.Count);
        var parts = new List<string>();
        foreach (var (token, rank) in ranks.OrderBy(entry => entry.Value))
        {
            if (token.Length == 1)
            {
                continue;
            }

            parts.Clear();
            foreach (char c in token)
            {
                parts.Add(c.ToString());
            }

            while (true)
            {
                int best = -1;
                int bestRank = int.MaxValue;
                for (int i = 0; i + 1 < parts.Count; i++)
                {
                    if (ranks.TryGetValue(parts[i] + parts[i + 1], out int pairRank) && pairRank < bestRank)
                    {
                        best = i;
                        bestRank = pairRank;
                    }
                }

                if (best < 0 || bestRank >= rank)
                {
                    break;
                }

                parts[best] += parts[best + 1];
                parts.RemoveAt(best + 1);
            }

            if (parts.Count == 2)
            {
                merges.Add(Spelling(parts[0], spellings) + " " + Spelling(parts[1], spellings));
            }
        }

        return [.. merges];
    }

    private static string Spelling(string bytes, Dictionary<string, string> spellings)
        => spellings.TryGetValue(bytes, out var spelling) ? spelling : Spell(bytes, ByteToUnicode());

    /// <summary>The GPT-2 byte-to-unicode spelling of a Latin-1 byte string.</summary>
    private static string Spell(string bytes, char[] byteChars)
    {
        var builder = new StringBuilder(bytes.Length);
        foreach (char b in bytes)
        {
            builder.Append(byteChars[b]);
        }

        return builder.ToString();
    }

    /// <summary>GPT-2's byte-to-unicode table: printable bytes map to themselves, the others to U+0100 onwards.</summary>
    private static char[] ByteToUnicode()
    {
        var map = new char[256];
        int next = 0;
        for (int b = 0; b < 256; b++)
        {
            bool printable = b is >= '!' and <= '~' or >= 0xA1 and <= 0xAC or >= 0xAE and <= 0xFF;
            map[b] = printable ? (char)b : (char)(256 + next++);
        }

        return map;
    }
}
