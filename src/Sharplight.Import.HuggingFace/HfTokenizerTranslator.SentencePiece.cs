using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

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
        md[CanonicalKeys.TokenizerAddSpacePrefix] = model.AddDummyPrefix || NormalizerPrependsSpace(tok);
        // Non-legacy tokenizer.json (Mistral v0.2, Llama 2 conversions with legacy = false): the dummy prefix comes from a Metaspace
        // pre-tokenizer, which skips text that already starts with a space and, with prepend_scheme "first", text after a special token.
        string? scheme = MetaspacePrependScheme(tok);
        if (!NormalizerPrependsSpace(tok) && scheme is "first" or "always" && md[CanonicalKeys.TokenizerAddSpacePrefix] is true)
        {
            md[CanonicalKeys.TokenizerSpacePrefixScheme] = scheme;
        }
        HfSpecialVocab.AddTo(md, dir, config, tok, ReadTokenizerConfig(dir), addedTokens);
    }

    /// <summary>The <c>prepend_scheme</c> of tokenizer.json's Metaspace pre-tokenizer, or null when it has none.</summary>
    private static string? MetaspacePrependScheme(JsonElement tok)
    {
        if (!tok.TryGetProperty("pre_tokenizer", out var preTokenizer) || preTokenizer.ValueKind != JsonValueKind.Object
            || FindMetaspace(preTokenizer) is not { } metaspace)
        {
            return null;
        }

        return metaspace.TryGetProperty("prepend_scheme", out var scheme) ? scheme.GetString() : null;
    }

    /// <summary>
    /// Whether tokenizer.json's normalizer prepends ▁ (transformers' legacy Llama tokenizer): it then applies even when
    /// <c>tokenizer.model</c> turns its dummy prefix off (MiniCPM3).
    /// </summary>
    private static bool NormalizerPrependsSpace(JsonElement tok)
    {
        if (!tok.TryGetProperty("normalizer", out var normalizer) || normalizer.ValueKind != JsonValueKind.Object)
        {
            return false;
        }

        var steps = normalizer.TypeName() == "Sequence" && normalizer.TryGetProperty("normalizers", out var children)
            ? [.. children.EnumerateArray()]
            : new[] { normalizer };
        return steps.Any(step => step.TypeName() == "Prepend" && step.TryGetProperty("prepend", out var text) && text.GetString() == "▁");
    }
}
