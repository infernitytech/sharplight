using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

// SentencePiece-style BPE (Llama 2, Mistral, Gemma, Phi-3): the vocabulary lives in tokenizer.model.
internal static partial class HfTokenizerTranslator
{
    /// <summary>
    /// SentencePiece vocabulary (llama.cpp <c>_create_vocab_sentencepiece</c>): pieces, scores and types from
    /// <c>tokenizer.model</c>, extra added tokens appended into unused slots, padded to the model's vocab_size.
    /// </summary>
    private static void AddSentencePiece(IDictionary<string, object> md, string dir, HfConfig config, JsonElement tok)
    {
        var model = SentencePieceModelFile.Read(Path.Combine(dir, "tokenizer.model"));
        int vocabSize = config.GetInt("vocab_size") ?? model.Pieces.Length;
        var tokens = new string[vocabSize];
        var scores = new float[vocabSize];
        var types = new int[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            if (i < model.Pieces.Length)
            {
                tokens[i] = model.Pieces[i];
                scores[i] = model.Scores[i];
                types[i] = model.Types[i];
            }
            else
            {
                tokens[i] = $"[PAD{i}]";
                scores[i] = PlaceholderScore;
                types[i] = TypeUnused;
            }
        }

        var addedTokens = tok.ArrayOrEmpty("added_tokens");
        var existing = new HashSet<string>(tokens, StringComparer.Ordinal);
        foreach (var added in addedTokens)
        {
            int id = added.GetProperty("id").GetInt32();
            string content = added.GetProperty("content").GetString()!;
            if (id >= vocabSize || existing.Contains(content) || types[id] != TypeUnused)
            {
                continue;   // only fills unused slots
            }

            tokens[id] = content;
            scores[id] = PlaceholderScore;
            types[id] = added.IsTrue("special") ? TypeControl : TypeUserDefined;
        }

        md[CanonicalKeys.TokenizerModel] = "spm";
        md[CanonicalKeys.TokenizerPre] = "default";
        md[CanonicalKeys.TokenizerTokens] = tokens;
        md[CanonicalKeys.TokenizerScores] = scores;
        md[CanonicalKeys.TokenizerTokenTypes] = types;
        md[CanonicalKeys.TokenizerAddSpacePrefix] = model.AddDummyPrefix;
        HfSpecialVocab.AddTo(md, dir, config, tok, ReadTokenizerConfig(dir), addedTokens);
    }
}
