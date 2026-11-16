using System.Text;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

internal static partial class HfTokenizerTranslator
{
    /// <summary>Whether tokenizer.json is a character-level BPE behind tokenizers' <c>Whitespace</c> pre-tokenizer (no byte-level steps; Chatterbox).</summary>
    private static bool IsWhitespaceBpe(JsonElement tok, JsonElement model)
        => !model.IsTrue("byte_fallback") && tok.TryGetProperty("pre_tokenizer", out var pre) && pre.ValueKind == JsonValueKind.Object
           && pre.TryGetProperty("type", out var type) && type.GetString() == "Whitespace";

    /// <summary>
    /// A character-level BPE over whitespace-split words as the byte-level form the engine runs: every token and merge spelled in GPT-2's
    /// byte-to-unicode alphabet over its UTF-8 bytes, and for every multi-byte character of the vocabulary merges ranked before all others that rebuild
    /// it from its bytes, so the byte-level merging reproduces the character-level one. The pre-tokenizer is <c>whitespace</c>.
    /// </summary>
    private static void AddWhitespaceBpe(IDictionary<string, object> md, string dir, HfConfig config, JsonElement tok, JsonElement model)
    {
        var tokenizerConfig = ReadTokenizerConfig(dir);
        var addedTokens = tok.ArrayOrEmpty("added_tokens");
        var addedById = new Dictionary<int, JsonElement>();
        var vocab = ReadBpeVocabulary(model, addedTokens, tokenizerConfig, addedById);
        int vocabSize = config.GetInt("vocab_size") ?? (vocab.Count == 0 ? 0 : vocab.Values.Max() + 1);
        var (tokens, types) = BuildBpeTokens(vocab, addedById, vocabSize, spaceBpe: false);
        var byteChars = ByteToUnicode();
        string Respell(string text) => Spell(Encoding.Latin1.GetString(Encoding.UTF8.GetBytes(text)), byteChars);
        var characterMerges = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
        {
            if (addedById.ContainsKey(i) || !vocab.ContainsKey(tokens[i]))
            {
                continue;
            }

            foreach (var rune in tokens[i].EnumerateRunes())
            {
                var bytes = Encoding.UTF8.GetBytes(rune.ToString());
                if (bytes.Length < 2 || !seen.Add(rune.ToString()))
                {
                    continue;
                }

                for (int b = 1; b < bytes.Length; b++)
                {
                    characterMerges.Add(Spell(Encoding.Latin1.GetString(bytes, 0, b), byteChars) + " " + Spell(Encoding.Latin1.GetString(bytes, b, 1), byteChars));
                }
            }

            tokens[i] = Respell(tokens[i]);
        }

        var merges = (ReadMerges(model) ?? [])
            .Select(merge => merge.Split(' ', 2))
            .Where(parts => parts.Length == 2)
            .Select(parts => Respell(parts[0]) + " " + Respell(parts[1]));
        md[CanonicalKeys.TokenizerModel] = "gpt2";
        md[CanonicalKeys.TokenizerPre] = "whitespace";
        md[CanonicalKeys.TokenizerTokens] = tokens;
        md[CanonicalKeys.TokenizerTokenTypes] = types;
        md[CanonicalKeys.TokenizerMerges] = (string[])[.. characterMerges, .. merges];
        HfSpecialVocab.AddTo(md, dir, config, tok, tokenizerConfig, addedTokens);
    }
}
