using System.Text.Json;

namespace Sharplight.HuggingFace;

/// <summary>
/// Translates a Hugging Face tokenizer (<c>tokenizer.json</c>, <c>tokenizer_config.json</c>, <c>config.json</c>,
/// chat template files) into the canonical <c>tokenizer.*</c> metadata, following llama.cpp's
/// <c>TextModel._set_vocab_gpt2</c> / <c>get_vocab_base</c> and gguf-py's <c>SpecialVocab</c>.
/// Supported model types: byte-level BPE ("gpt2"), SentencePiece-style BPE with byte fallback, WordPiece and Unigram.
/// The per-model work lives in the partial files <c>.Bpe</c>, <c>.SentencePiece</c> and <c>.Vocabulary</c>; special tokens in
/// <see cref="HfSpecialVocab"/>.
/// </summary>
internal static partial class HfTokenizerTranslator
{
    // llama.cpp token types (gguf-py TokenType).
    private const int TypeNormal = 1;
    private const int TypeControl = 3;
    private const int TypeUserDefined = 4;
    private const int TypeUnused = 5;
    private const int TypeByte = 6;

    /// <summary>Score of tokens that have no meaningful probability (padding and added tokens).</summary>
    private const float PlaceholderScore = -1000f;

    /// <summary>Adds the canonical <c>tokenizer.*</c> keys of the checkpoint in <paramref name="dir"/> to <paramref name="md"/>.</summary>
    /// <param name="md">The canonical metadata being built.</param>
    /// <param name="dir">The checkpoint directory (holds <c>tokenizer.json</c> and optional companions).</param>
    /// <param name="config">The checkpoint's <c>config.json</c>; supplies <c>vocab_size</c> and special-token id fallbacks.</param>
    /// <exception cref="NotSupportedException">There is no <c>tokenizer.json</c> or its model type is not supported.</exception>
    public static void AddTo(IDictionary<string, object> md, string dir, HfConfig config)
    {
        string tokenizerPath = Path.Combine(dir, "tokenizer.json");
        if (!File.Exists(tokenizerPath))
        {
            throw new NotSupportedException($"{dir}: no tokenizer.json (SentencePiece-only tokenizers are not supported yet).");
        }

        using var tokenizerDocument = JsonDocument.Parse(File.ReadAllBytes(tokenizerPath));
        var tok = tokenizerDocument.RootElement;

        var model = tok.GetProperty("model");
        string modelType = model.TypeName();
        if (modelType.Length == 0)
        {
            // Files written by older tokenizers releases (e.g. facebook/bart-base) leave the type out; the model's fields tell it.
            modelType = model.TryGetProperty("merges", out _) ? "BPE"
                : model.TryGetProperty("max_input_chars_per_word", out _) ? "WordPiece"
                : model.TryGetProperty("vocab", out var vocab) && vocab.ValueKind == JsonValueKind.Array ? "Unigram" : "";
        }

        // SentencePiece-style (Llama 2, Mistral, Gemma, Phi-3): BPE with byte fallback, scores come from tokenizer.model.
        if (modelType == "BPE" && model.IsTrue("byte_fallback") && File.Exists(Path.Combine(dir, "tokenizer.model")))
        {
            AddSentencePiece(md, dir, config, tok);
            return;
        }

        if (modelType is "WordPiece" or "Unigram")
        {
            AddVocabularyModel(md, dir, config, tok, model, modelType);
            return;
        }

        if (modelType != "BPE")
        {
            throw new NotSupportedException($"tokenizer.json model type '{modelType}' is not supported (BPE, WordPiece, Unigram).");
        }

        AddByteLevelBpe(md, dir, config, tok, model);
    }

    /// <summary>The parsed <c>tokenizer_config.json</c> (detached from its document), or null when absent.</summary>
    private static JsonElement? ReadTokenizerConfig(string dir)
    {
        string path = Path.Combine(dir, "tokenizer_config.json");
        if (!File.Exists(path))
        {
            return null;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        return document.RootElement.Clone();
    }
}
