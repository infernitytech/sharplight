using System.Text;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Import.WhisperCpp;

/// <summary>
/// The vocabulary of a whisper.cpp model in Sharplight's canonical form. The file stores only the text tokens as byte strings (the
/// ranks of Whisper's byte-level BPE); the special tokens follow from the vocabulary size as whisper.cpp derives them. This class
/// rebuilds what the canonical metadata needs: the token strings in the byte-level alphabet of GPT-2-style BPE, the merges (recovered
/// from the ranks: each token's two parts are what byte-pair encoding of the token stops at below its own rank), the special tokens under
/// their Whisper names, and the speech-protocol keys (<see cref="SpeechKeys"/>) including the non-speech tokens Whisper suppresses.
/// </summary>
internal static class WhisperCppVocabulary
{
    /// <summary>Single symbols and symbol runs that Whisper never generates (its <c>non_speech_tokens</c>).</summary>
    private static readonly string[] s_nonSpeechSymbols =
    [
        "\"", "#", "(", ")", "*", "+", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "「", "」", "『", "』",
        "<<", ">>", "<<<", ">>>", "--", "---", "-(", "-[", "('", "(\"", "((", "))", "(((", ")))", "[[", "]]", "{{", "}}", "♪♪", "♪♪♪",
    ];

    /// <summary>Musical symbols whose first token is suppressed even when they take several tokens.</summary>
    private static readonly string[] s_musicSymbols = ["♩", "♪", "♫", "♬", "♭", "♮", "♯"];

    /// <summary>Adds the canonical tokenizer keys and the speech-protocol keys of <paramref name="file"/> to <paramref name="metadata"/>.</summary>
    /// <param name="file">The whisper.cpp model.</param>
    /// <param name="metadata">The canonical metadata being built.</param>
    public static void AddTo(WhisperCppFile file, IDictionary<string, object> metadata)
    {
        var hp = file.HyperParameters;
        var ids = new WhisperCppSpecialTokens(hp);
        var alphabet = ByteAlphabet();
        int vocabulary = hp.VocabSize;
        var tokens = new string[vocabulary];
        var types = new int[vocabulary];
        for (int id = 0; id < vocabulary; id++)
        {
            bool text = id < file.Tokens.Count && id < ids.EndOfText && file.Tokens[id].Length > 0;
            tokens[id] = text ? ToAlphabet(file.Tokens[id], alphabet) : ids.Name(id);
            types[id] = (int)(text ? TokenType.Normal : TokenType.Control);
        }

        var data = new TokenizerData
        {
            Model = "gpt2",
            PreTokenizer = "gpt-2",
            Tokens = tokens,
            TokenTypes = types,
            Merges = Merges(tokens, ids.EndOfText),
            EosId = ids.EndOfText,
            EosIds = [ids.EndOfText],
            BosId = ids.EndOfText,
            AddBos = false,
        };
        metadata[CanonicalKeys.TokenizerModel] = data.Model;
        metadata[CanonicalKeys.TokenizerPre] = data.PreTokenizer;
        metadata[CanonicalKeys.TokenizerTokens] = data.Tokens;
        metadata[CanonicalKeys.TokenizerTokenTypes] = data.TokenTypes;
        metadata[CanonicalKeys.TokenizerMerges] = data.Merges!;
        metadata[CanonicalKeys.TokenizerEosId] = ids.EndOfText;
        metadata[CanonicalKeys.TokenizerBosId] = ids.EndOfText;
        metadata[CanonicalKeys.TokenizerAddBos] = false;

        metadata[SpeechKeys.StartId] = ids.StartOfTranscript;
        metadata[SpeechKeys.EndId] = ids.EndOfText;
        metadata[SpeechKeys.NoTimestampsId] = ids.NoTimestamps;
        metadata[SpeechKeys.TimestampBeginId] = ids.TimestampBegin;
        metadata[SpeechKeys.TimestampStep] = 0.02f;
        metadata[SpeechKeys.MaxTextTokens] = hp.TextContext / 2;
        metadata[SpeechKeys.MaxInitialTimestampIndex] = 50;
        metadata[SpeechKeys.PreviousId] = ids.StartOfPrevious;
        metadata[SpeechKeys.NoSpeechId] = ids.NoSpeech;
        if (hp.IsMultilingual)
        {
            metadata[SpeechKeys.Languages] = WhisperCppSpecialTokens.Languages.Take(hp.LanguageCount).ToArray();
            metadata[SpeechKeys.LanguageIds] = Enumerable.Range(ids.StartOfTranscript + 1, hp.LanguageCount).ToArray();
            metadata[SpeechKeys.TranscribeId] = ids.Transcribe;
            metadata[SpeechKeys.TranslateId] = ids.Translate;
        }

        var tokenizer = TokenizerFactory.Create(data);
        metadata[SpeechKeys.SuppressIds] = NonSpeechTokens(tokenizer)
            .Concat([ids.Translate, ids.Transcribe, ids.StartOfTranscript, ids.StartOfPrevious, ids.StartOfLanguageModel, ids.NoSpeech])
            .Distinct().Order().ToArray();
        metadata[SpeechKeys.BeginSuppressIds] = new[] { tokenizer.Encode(" ", addSpecial: false, parseSpecial: false)[0], ids.EndOfText };
    }

    /// <summary>Whisper's non-speech tokens: symbols that alone (or after a space) are one token, the first token of musical symbols, " -" and " '".</summary>
    private static IEnumerable<int> NonSpeechTokens(ITokenizer tokenizer)
    {
        int[] Encode(string text) => tokenizer.Encode(text, addSpecial: false, parseSpecial: false);
        var result = new SortedSet<int> { Encode(" -")[0], Encode(" '")[0] };
        foreach (string symbol in s_nonSpeechSymbols.Concat(s_musicSymbols))
        {
            bool music = s_musicSymbols.Contains(symbol);
            foreach (var encoded in new[] { Encode(symbol), Encode(" " + symbol) })
            {
                if (encoded.Length == 1 || music)
                {
                    result.Add(encoded[0]);
                }
            }
        }

        return result;
    }

    /// <summary>The merges in rank order, recovered from the token ranks (ids) of the text tokens below <paramref name="textTokens"/>.</summary>
    private static string[] Merges(string[] tokens, int textTokens)
    {
        var ranks = new Dictionary<string, int>(textTokens, StringComparer.Ordinal);
        for (int id = 0; id < textTokens; id++)
        {
            ranks.TryAdd(tokens[id], id);
        }

        var merges = new List<string>(textTokens);
        var parts = new List<string>();
        for (int id = 0; id < textTokens; id++)
        {
            string token = tokens[id];
            if (token.Length < 2 || ranks[token] != id)
            {
                continue;
            }

            parts.Clear();
            foreach (char c in token)
            {
                parts.Add(c.ToString());
            }

            while (parts.Count > 2)
            {
                int best = -1;
                int bestRank = int.MaxValue;
                for (int i = 0; i + 1 < parts.Count; i++)
                {
                    if (ranks.TryGetValue(parts[i] + parts[i + 1], out int rank) && rank < bestRank)
                    {
                        best = i;
                        bestRank = rank;
                    }
                }

                if (best < 0 || bestRank >= id)
                {
                    break;
                }

                parts[best] += parts[best + 1];
                parts.RemoveAt(best + 1);
            }

            if (parts.Count == 2)
            {
                merges.Add(parts[0] + " " + parts[1]);
            }
        }

        return [.. merges];
    }

    /// <summary>GPT-2's byte-level alphabet: printable bytes map to themselves, the others to code points from U+0100 upwards.</summary>
    private static char[] ByteAlphabet()
    {
        var alphabet = new char[256];
        var assigned = new bool[256];
        foreach (var (first, last) in new[] { ('!', '~'), ('¡', '¬'), ('®', 'ÿ') })
        {
            for (int b = first; b <= last; b++)
            {
                alphabet[b] = (char)b;
                assigned[b] = true;
            }
        }

        int next = 256;
        for (int b = 0; b < 256; b++)
        {
            if (!assigned[b])
            {
                alphabet[b] = (char)next++;
            }
        }

        return alphabet;
    }

    private static string ToAlphabet(byte[] bytes, char[] alphabet)
    {
        var builder = new StringBuilder(bytes.Length);
        foreach (byte b in bytes)
        {
            builder.Append(alphabet[b]);
        }

        return builder.ToString();
    }
}
