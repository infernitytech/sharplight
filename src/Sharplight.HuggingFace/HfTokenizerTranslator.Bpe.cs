using System.Globalization;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

// Byte-level BPE ("gpt2") and character-level BPE with byte fallback ("bpe") vocabularies.
internal static partial class HfTokenizerTranslator
{
    /// <summary>Known pre-tokenizer split regexes (as found in tokenizer.json) → Sharplight/llama.cpp preset names.</summary>
    private static readonly (string Preset, string Regex)[] s_knownRegexes =
    [
        ("qwen35", @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        ("qwen2", @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        ("llama-bpe", @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        ("pixtral", @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        ("llama4", @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        ("gpt-2", @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"),
        // Seed-Coder / Seed-OSS write the CR and LF characters themselves into the regex (not as escapes).
        ("seed-coder", "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\r\n]+|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"),
    ];

    private static void AddByteLevelBpe(IDictionary<string, object> md, string dir, HfConfig config, JsonElement tok, JsonElement model)
    {
        var tokenizerConfig = ReadTokenizerConfig(dir);
        var addedTokens = tok.ArrayOrEmpty("added_tokens");

        // Gemma 4: character-level BPE with merge ranks and byte fallback over a space → ▁ normalizer (no tokenizer.model, no pre-tokenizer regex).
        bool spaceBpe = model.IsTrue("byte_fallback")
            && tok.TryGetProperty("normalizer", out var normalizer) && normalizer.ValueKind == JsonValueKind.Object
            && normalizer.TryGetProperty("type", out var normalizerType) && normalizerType.GetString() == "Replace"
            && normalizer.TryGetProperty("content", out var normalizerContent) && normalizerContent.GetString() == "▁";

        // ---- tokens + types (llama.cpp get_vocab_base) ---------------------------------------
        var addedById = new Dictionary<int, JsonElement>();
        var vocab = ReadBpeVocabulary(model, addedTokens, tokenizerConfig, addedById);

        int maxId = vocab.Count == 0 ? -1 : vocab.Values.Max();
        int vocabSize = config.GetInt("vocab_size") ?? vocab.Count;
        // Tokens beyond the model's embedding table cannot be produced or fed (e.g. Gemma 3's extra added token): drop them.
        if (maxId >= vocabSize)
        {
            foreach (var token in vocab.Where(entry => entry.Value >= vocabSize).Select(entry => entry.Key).ToList())
            {
                vocab.Remove(token);
            }
        }

        var (tokens, types) = BuildBpeTokens(vocab, addedById, vocabSize, spaceBpe);

        md[CanonicalKeys.TokenizerModel] = spaceBpe ? "bpe" : "gpt2";
        md[CanonicalKeys.TokenizerPre] = spaceBpe ? "default" : DetectPreTokenizer(tok);
        md[CanonicalKeys.TokenizerTokens] = tokens;
        md[CanonicalKeys.TokenizerTokenTypes] = types;
        if (ReadMerges(model) is { Length: > 0 } merges)
        {
            md[CanonicalKeys.TokenizerMerges] = merges;
        }

        HfSpecialVocab.AddTo(md, dir, config, tok, tokenizerConfig, addedTokens);
    }

    /// <summary>
    /// The token → id map transformers exposes: the model vocabulary plus tokenizer.json's added tokens, plus tokens only listed in
    /// tokenizer_config.json's <c>added_tokens_decoder</c>. <paramref name="addedById"/> receives the added-token entries by id.
    /// </summary>
    private static Dictionary<string, int> ReadBpeVocabulary(JsonElement model, JsonElement[] addedTokens, JsonElement? tokenizerConfig, Dictionary<int, JsonElement> addedById)
    {
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var entry in model.GetProperty("vocab").EnumerateObject())
        {
            vocab[entry.Name] = entry.Value.GetInt32();
        }

        foreach (var added in addedTokens)
        {
            int id = added.GetProperty("id").GetInt32();
            vocab[added.GetProperty("content").GetString()!] = id; // transformers' vocab includes added tokens
            addedById[id] = added;
        }

        // transformers also merges tokenizer_config.json's added_tokens_decoder (it may list tokens tokenizer.json lacks).
        if (tokenizerConfig is { ValueKind: JsonValueKind.Object } config
            && config.TryGetProperty("added_tokens_decoder", out var decoder) && decoder.ValueKind == JsonValueKind.Object)
        {
            foreach (var entry in decoder.EnumerateObject())
            {
                if (!int.TryParse(entry.Name, NumberStyles.None, CultureInfo.InvariantCulture, out int id) || addedById.ContainsKey(id))
                {
                    continue;
                }

                if (entry.Value.TryGetProperty("content", out var content) && content.GetString() is { } contentText)
                {
                    vocab[contentText] = id;
                    addedById[id] = entry.Value;
                }
            }
        }

        return vocab;
    }

    /// <summary>Tokens and llama.cpp token types in id order; ids without a token become unused <c>[PAD{i}]</c> slots.</summary>
    private static (string[] Tokens, int[] Types) BuildBpeTokens(Dictionary<string, int> vocab, Dictionary<int, JsonElement> addedById, int vocabSize, bool spaceBpe)
    {
        var tokenById = new Dictionary<int, string>(vocab.Count);
        foreach (var (token, id) in vocab)
        {
            tokenById[id] = token;
        }

        var tokens = new string[vocabSize];
        var types = new int[vocabSize];
        var addedContents = new HashSet<string>(addedById.Values.Select(a => a.GetProperty("content").GetString()!), StringComparer.Ordinal);
        for (int i = 0; i < vocabSize; i++)
        {
            if (!tokenById.TryGetValue(i, out var token))
            {
                tokens[i] = $"[PAD{i}]";
                types[i] = TypeUnused;
                continue;
            }

            if (addedContents.Contains(token) && addedById.TryGetValue(i, out var added))
            {
                // TODO: llama.cpp re-normalizes non-"normalized" added tokens via decode(encode(tok)); identity for common vocabularies.
                if (added.IsTrue("special") || LooksSpecial(token))
                {
                    types[i] = TypeControl;
                }
                else
                {
                    if (!spaceBpe)
                    {
                        token = token.Replace("▁", " ");
                    }

                    types[i] = TypeUserDefined;
                }
            }
            else
            {
                types[i] = spaceBpe && IsByteToken(token) ? TypeByte : TypeNormal;
            }

            tokens[i] = token;
        }

        return (tokens, types);
    }

    private static bool IsByteToken(string token) => token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token[5] == '>';

    private static bool LooksSpecial(string token)
        => token is "<pad>" or "<mask>" or "<2mass>" or "[@BOS@]"
           || (token.StartsWith("<|", StringComparison.Ordinal) && token.EndsWith("|>", StringComparison.Ordinal))
           || (token.StartsWith("<｜", StringComparison.Ordinal) && token.EndsWith("｜>", StringComparison.Ordinal))
           || (token.StartsWith("<unused", StringComparison.Ordinal) && token.EndsWith('>'));

    // ---- pre-tokenizer ------------------------------------------------------------------------

    /// <summary>The llama.cpp pre-tokenizer preset matching tokenizer.json's pre-tokenizer.</summary>
    /// <exception cref="NotSupportedException">The split regexes match no known preset.</exception>
    private static string DetectPreTokenizer(JsonElement tok)
    {
        var regexes = new List<string>();
        bool byteLevelRegex = false;
        if (tok.TryGetProperty("pre_tokenizer", out var preTokenizer) && preTokenizer.ValueKind == JsonValueKind.Object)
        {
            CollectSplitRegexes(preTokenizer, regexes, ref byteLevelRegex);
        }

        if (byteLevelRegex && regexes.Count == 0)
        {
            return "gpt-2";
        }

        // Flint: three successive splits (digit groups, CJK runs, case-aware words).
        if (regexes.Count == 3 && regexes[0] == @"\p{N}{1,3}" && regexes[1].StartsWith("[一", StringComparison.Ordinal))
        {
            return "flint";
        }

        foreach (var regex in regexes)
        {
            foreach (var (preset, known) in s_knownRegexes)
            {
                if (regex == known)
                {
                    return preset;
                }
            }
        }

        throw new NotSupportedException(
            "Unknown BPE pre-tokenizer; add its split regex to HfTokenizerTranslator.s_knownRegexes. Regexes found: " +
            (regexes.Count == 0 ? "(none)" : string.Join(" || ", regexes)));
    }

    /// <summary>Collects the <c>Split</c> regexes of a (possibly nested) pre-tokenizer and whether a ByteLevel step splits with its own regex.</summary>
    private static void CollectSplitRegexes(JsonElement preTokenizer, List<string> regexes, ref bool byteLevelRegex)
    {
        string type = preTokenizer.TypeName();
        if (type == "Sequence" && preTokenizer.TryGetProperty("pretokenizers", out var children))
        {
            foreach (var child in children.EnumerateArray())
            {
                CollectSplitRegexes(child, regexes, ref byteLevelRegex);
            }
        }
        else if (type == "Split" && preTokenizer.TryGetProperty("pattern", out var pattern) && pattern.TryGetProperty("Regex", out var regex))
        {
            regexes.Add(regex.GetString() ?? "");
        }
        else if (type == "ByteLevel" && (!preTokenizer.TryGetProperty("use_regex", out var useRegex) || useRegex.ValueKind != JsonValueKind.False))
        {
            byteLevelRegex = true;
        }
    }

    // ---- merges ---------------------------------------------------------------------------------
    private static string[]? ReadMerges(JsonElement model)
    {
        if (!model.TryGetProperty("merges", out var merges) || merges.ValueKind != JsonValueKind.Array)
        {
            return null;
        }

        var list = new string[merges.GetArrayLength()];
        int i = 0;
        foreach (var merge in merges.EnumerateArray())
        {
            if (merge.ValueKind == JsonValueKind.String)
            {
                list[i++] = merge.GetString()!;
            }
            else if (merge.ValueKind == JsonValueKind.Array && merge.GetArrayLength() == 2)
            {
                // transformers >= 4.45 pair format; spaces inside a part are encoded as U+0120 (chr(ord(' ') + 256))
                list[i++] = EncodeSpaces(merge[0].GetString()!) + " " + EncodeSpaces(merge[1].GetString()!);
            }
            else
            {
                throw new InvalidDataException("Unknown tokenizer merges format.");
            }
        }

        return list;
    }

    private static string EncodeSpaces(string part) => part.Contains(' ') ? part.Replace(' ', 'Ġ') : part;
}
