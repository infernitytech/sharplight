using System.Text;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

internal static partial class HfTokenizerTranslator
{
    /// <summary>The tokenizer file of Mistral checkpoints released in their own format (Voxtral, Ministral, Pixtral).</summary>
    private const string TekkenFile = "tekken.json";

    /// <summary>Whether <paramref name="dir"/> holds a tekken tokenizer.</summary>
    private static bool IsTekken(string dir) => File.Exists(Path.Combine(dir, TekkenFile));

    /// <summary>
    /// Byte-level BPE metadata of a tekken tokenizer (mistral-common's tiktoken format): the special tokens take the first ids (control tokens
    /// when the file says so, user-defined otherwise), the mergeable ranks follow them in GPT-2 byte-to-unicode spelling up to the vocabulary
    /// size, merges are recovered from the ranks as for other tiktoken vocabularies, and the pre-tokenizer is <c>tekken</c>. Begin-of-sequence
    /// <c>&lt;s&gt;</c> is added in front of prompts, as mistral-common does.
    /// </summary>
    private static void AddTekken(IDictionary<string, object> md, string dir, HfConfig config)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, TekkenFile)));
        var root = document.RootElement;
        var settings = root.GetProperty("config");
        var specials = root.GetProperty("special_tokens");
        int specialCount = settings.TryGetProperty("default_num_special_tokens", out var n) ? n.GetInt32() : specials.GetArrayLength();
        int vocabSize = config.GetInt("vocab_size") ?? (settings.TryGetProperty("default_vocab_size", out var v) ? v.GetInt32() : 131072);
        var byteChars = ByteToUnicode();
        var tokens = new string[vocabSize];
        var types = new int[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            tokens[i] = $"<SPECIAL_{i}>";
            types[i] = TypeControl;
        }

        int bos = 1;
        int eos = 2;
        int unk = 0;
        int? pad = null;
        foreach (var special in specials.EnumerateArray())
        {
            int rank = special.GetProperty("rank").GetInt32();
            if (rank >= vocabSize)
            {
                continue;
            }

            string text = special.GetProperty("token_str").GetString() ?? $"<SPECIAL_{rank}>";
            tokens[rank] = text;
            types[rank] = special.TryGetProperty("is_control", out var control) && control.ValueKind == JsonValueKind.False ? TypeUserDefined : TypeControl;
            switch (text)
            {
                case "<s>":
                    bos = rank;
                    break;
                case "</s>":
                    eos = rank;
                    break;
                case "<unk>":
                    unk = rank;
                    break;
                case "<pad>":
                    pad = rank;
                    break;
            }
        }

        var ranks = new Dictionary<string, int>(StringComparer.Ordinal);
        var spellings = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var entry in root.GetProperty("vocab").EnumerateArray())
        {
            int rank = entry.GetProperty("rank").GetInt32();
            int id = specialCount + rank;
            if (id >= vocabSize)
            {
                continue;
            }

            string bytes = Encoding.Latin1.GetString(Convert.FromBase64String(entry.GetProperty("token_bytes").GetString()!));
            string spelling = Spell(bytes, byteChars);
            ranks[bytes] = rank;
            spellings[bytes] = spelling;
            tokens[id] = spelling;
            types[id] = TypeNormal;
        }

        md[CanonicalKeys.TokenizerModel] = "gpt2";
        md[CanonicalKeys.TokenizerPre] = "tekken";
        md[CanonicalKeys.TokenizerTokens] = tokens;
        md[CanonicalKeys.TokenizerTokenTypes] = types;
        md[CanonicalKeys.TokenizerMerges] = TiktokenMerges(ranks, spellings);
        md[CanonicalKeys.TokenizerBosId] = bos;
        md[CanonicalKeys.TokenizerEosId] = eos;
        md[CanonicalKeys.TokenizerUnkId] = unk;
        if (pad is { } padId)
        {
            md[CanonicalKeys.TokenizerPadId] = padId;
        }

        md[CanonicalKeys.TokenizerAddBos] = true;
        md[CanonicalKeys.TokenizerAddEos] = false;
    }
}
