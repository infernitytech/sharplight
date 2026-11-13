using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

// WordPiece (BERT family) and Unigram (SentencePiece unigram, XLM-RoBERTa family) vocabularies.
internal static partial class HfTokenizerTranslator
{
    /// <summary>Exclusive upper bound of a plausible <c>model_max_length</c>; transformers uses huge sentinels for "unlimited".</summary>
    private const long MaxPlausibleModelLength = 1_000_000;

    /// <summary>
    /// WordPiece (BERT family) and Unigram (SentencePiece unigram, XLM-RoBERTa family) vocabularies: tokens in id order (Hugging
    /// Face form: WordPiece continuations keep their <c>##</c>, unigram pieces their ▁), unigram scores, normalizer flags and the
    /// pair layout of cross-encoders.
    /// </summary>
    private static void AddVocabularyModel(IDictionary<string, object> md, string dir, HfConfig config, JsonElement tok, JsonElement model, string modelType)
    {
        bool isWordPiece = modelType == "WordPiece";
        var addedTokens = tok.ArrayOrEmpty("added_tokens");
        var pieces = new List<string>();
        var pieceScores = new List<float>();
        if (isWordPiece)
        {
            ReadWordPieceVocabulary(model, pieces);
        }
        else
        {
            foreach (var entry in model.GetProperty("vocab").EnumerateArray())
            {
                pieces.Add(entry[0].GetString()!);
                pieceScores.Add(entry[1].GetSingle());
            }

            if (model.TryGetProperty("unk_id", out var unkId) && unkId.ValueKind == JsonValueKind.Number)
            {
                md[CanonicalKeys.TokenizerUnkId] = (uint)unkId.GetInt32();
            }
        }

        int vocabSize = Math.Max(config.GetInt("vocab_size") ?? pieces.Count, pieces.Count);
        var tokens = new string[vocabSize];
        var types = new int[vocabSize];
        var scores = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            if (i < pieces.Count && pieces[i].Length > 0)
            {
                tokens[i] = pieces[i];
                types[i] = TypeNormal;
                scores[i] = i < pieceScores.Count ? pieceScores[i] : 0;
            }
            else
            {
                tokens[i] = $"[PAD{i}]";
                types[i] = TypeUnused;
                scores[i] = PlaceholderScore;
            }
        }

        foreach (var added in addedTokens)
        {
            int id = added.GetProperty("id").GetInt32();
            if (id >= vocabSize)
            {
                continue;
            }

            tokens[id] = added.GetProperty("content").GetString()!;
            types[id] = added.IsTrue("special") ? TypeControl : TypeUserDefined;
        }

        md[CanonicalKeys.TokenizerModel] = isWordPiece ? "wordpiece" : "unigram";
        md[CanonicalKeys.TokenizerPre] = "default";
        md[CanonicalKeys.TokenizerTokens] = tokens;
        md[CanonicalKeys.TokenizerTokenTypes] = types;
        if (!isWordPiece)
        {
            md[CanonicalKeys.TokenizerScores] = scores;
        }

        if (isWordPiece)
        {
            AddNormalizerFlags(md, tok);
        }
        else
        {
            md[CanonicalKeys.TokenizerAddSpacePrefix] = AddsSpacePrefix(tok);
        }

        AddPairStyle(md, tok);

        var tokenizerConfig = ReadTokenizerConfig(dir);
        if (tokenizerConfig is { ValueKind: JsonValueKind.Object } configRoot
            && configRoot.TryGetProperty("model_max_length", out var maxLength) && maxLength.ValueKind == JsonValueKind.Number
            && maxLength.TryGetInt64(out long maxLengthValue) && maxLengthValue is > 0 and < MaxPlausibleModelLength)
        {
            md[CanonicalKeys.TokenizerMaxLength] = (int)maxLengthValue;
        }

        HfSpecialVocab.AddTo(md, dir, config, tok, tokenizerConfig, addedTokens);
    }

    /// <summary>WordPiece stores a token → id map; this lays the tokens out in id order (gaps stay empty).</summary>
    private static void ReadWordPieceVocabulary(JsonElement model, List<string> pieces)
    {
        var vocab = model.GetProperty("vocab").EnumerateObject().Select(p => (p.Name, Id: p.Value.GetInt32())).ToList();
        int count = vocab.Count == 0 ? 0 : vocab.Max(v => v.Id) + 1;
        pieces.AddRange(Enumerable.Repeat("", count));
        foreach (var (name, id) in vocab)
        {
            pieces[id] = name;
        }
    }

    /// <summary>Normalizer flags (BertNormalizer: lowercase / strip_accents; a Lowercase step in a sequence).</summary>
    private static void AddNormalizerFlags(IDictionary<string, object> md, JsonElement tok)
    {
        bool lowercase = false;
        bool? stripAccents = null;
        if (tok.TryGetProperty("normalizer", out var normalizer) && normalizer.ValueKind == JsonValueKind.Object)
        {
            VisitNormalizer(normalizer, ref lowercase, ref stripAccents);
        }

        md[CanonicalKeys.TokenizerLowercase] = lowercase;
        if (stripAccents is { } strip)
        {
            md[CanonicalKeys.TokenizerStripAccents] = strip;
        }
    }

    private static void VisitNormalizer(JsonElement normalizer, ref bool lowercase, ref bool? stripAccents)
    {
        string type = normalizer.TypeName();
        if (type == "Sequence" && normalizer.TryGetProperty("normalizers", out var children))
        {
            foreach (var child in children.EnumerateArray())
            {
                VisitNormalizer(child, ref lowercase, ref stripAccents);
            }
        }
        else if (type == "BertNormalizer")
        {
            lowercase |= normalizer.IsTrue("lowercase");
            if (normalizer.TryGetProperty("strip_accents", out var strip) && strip.ValueKind is JsonValueKind.True or JsonValueKind.False)
            {
                stripAccents = strip.ValueKind == JsonValueKind.True;
            }
        }
        else if (type == "Lowercase")
        {
            lowercase = true;
        }
        else if (type == "StripAccents")
        {
            stripAccents = true;
        }
    }

    /// <summary>Whether the Metaspace pre-tokenizer prepends a space to the text (default true).</summary>
    private static bool AddsSpacePrefix(JsonElement tok)
    {
        if (!tok.TryGetProperty("pre_tokenizer", out var preTokenizer) || preTokenizer.ValueKind != JsonValueKind.Object
            || FindMetaspace(preTokenizer) is not { } metaspace)
        {
            return true;
        }

        return metaspace.TryGetProperty("prepend_scheme", out var scheme)
            ? scheme.GetString() != "never"
            : !metaspace.TryGetProperty("add_prefix_space", out var addPrefixSpace) || addPrefixSpace.ValueKind != JsonValueKind.False;
    }

    private static JsonElement? FindMetaspace(JsonElement preTokenizer)
    {
        string type = preTokenizer.TypeName();
        if (type == "Metaspace")
        {
            return preTokenizer;
        }

        if (type == "Sequence" && preTokenizer.TryGetProperty("pretokenizers", out var children))
        {
            foreach (var child in children.EnumerateArray())
            {
                if (FindMetaspace(child) is { } metaspace)
                {
                    return metaspace;
                }
            }
        }

        return null;
    }

    /// <summary>Pair layout: two separators between the texts (RoBERTa) or one with token type 1 for the second text (BERT).</summary>
    private static void AddPairStyle(IDictionary<string, object> md, JsonElement tok)
    {
        if (!tok.TryGetProperty("post_processor", out var processor) || processor.ValueKind != JsonValueKind.Object)
        {
            return;
        }

        string type = processor.TypeName();
        if (type == "RobertaProcessing")
        {
            md[CanonicalKeys.TokenizerPairStyle] = "roberta";
        }
        else if (type == "TemplateProcessing" && processor.TryGetProperty("pair", out var pair) && pair.ValueKind == JsonValueKind.Array)
        {
            var items = pair.EnumerateArray().ToList();
            int first = items.FindIndex(e => e.TryGetProperty("Sequence", out var sequence) && sequence.GetProperty("id").GetString() == "A");
            int second = items.FindIndex(e => e.TryGetProperty("Sequence", out var sequence) && sequence.GetProperty("id").GetString() == "B");
            if (first >= 0 && second > first)
            {
                md[CanonicalKeys.TokenizerPairStyle] = second - first - 1 >= 2 ? "roberta" : "bert";
            }
        }
    }
}
