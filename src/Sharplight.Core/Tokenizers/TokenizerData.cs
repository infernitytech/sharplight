using Sharplight.Models;

namespace Sharplight.Tokenizers;

/// <summary>
/// Format-agnostic description of a tokenizer: vocabulary, merges or scores, special-token ids and behaviour flags.
/// Build it by hand or via <see cref="FromMetadata"/> from canonical model metadata (<see cref="CanonicalKeys"/>).
/// Instances are immutable records; the arrays they hold are shared with the tokenizers built from them and must not be modified.
/// </summary>
public sealed record TokenizerData
{
    /// <summary>
    /// Tokenizer algorithm: "gpt2" (byte-level BPE), "spm" or "llama" (SentencePiece-style: scored merges, ▁ for spaces, byte fallback),
    /// "bpe" (character-level ranked BPE with byte fallback), "wordpiece" (BERT: greedy longest match, ## continuations) or
    /// "unigram" (SentencePiece unigram: Viterbi over piece scores).
    /// </summary>
    public required string Model { get; init; }

    /// <summary>Pre-tokenizer preset name of byte-level BPE, e.g. "qwen35", "qwen2", "llama-bpe", "gpt-2", "default". Null or empty means "default".</summary>
    public string? PreTokenizer { get; init; }

    /// <summary>The vocabulary: token string per id (ids are indices).</summary>
    public required string[] Tokens { get; init; }

    /// <summary>Per-token <see cref="TokenType"/> as ints. Null means all tokens are normal.</summary>
    public int[]? TokenTypes { get; init; }

    /// <summary>BPE merges, "left right" per entry, in rank order.</summary>
    public string[]? Merges { get; init; }

    /// <summary>Per-token scores of a SentencePiece ("spm") or unigram tokenizer; higher is better.</summary>
    public float[]? Scores { get; init; }

    /// <summary>SentencePiece and unigram: prepend a space to the text (default depends on the tokenizer model).</summary>
    public bool? AddSpacePrefix { get; init; }

    /// <summary>WordPiece: lowercase the text (default true).</summary>
    public bool? Lowercase { get; init; }

    /// <summary>WordPiece: strip accents (default: same as <see cref="Lowercase"/>).</summary>
    public bool? StripAccents { get; init; }

    /// <summary>Id of the beginning-of-sequence token, if any.</summary>
    public int? BosId { get; init; }

    /// <summary>The primary EOS id (used when <see cref="AddEos"/> is on).</summary>
    public int? EosId { get; init; }

    /// <summary>All ids that the metadata declares as end-of-generation (eos, eot, eom). Additional EOG tokens are detected by name.</summary>
    public int[] EosIds { get; init; } = [];

    /// <summary>Id of the padding token, if any.</summary>
    public int? PadId { get; init; }

    /// <summary>Id of the unknown token, if any (WordPiece and unigram fall back to a named or first token).</summary>
    public int? UnkId { get; init; }

    /// <summary>Whether to prepend BOS. Null = use the tokenizer model's default.</summary>
    public bool? AddBos { get; init; }

    /// <summary>Whether to append EOS. Null = the tokenizer model's default (false except for WordPiece).</summary>
    public bool? AddEos { get; init; }

    /// <summary>Canonical metadata key of <see cref="Model"/>.</summary>
    public const string KeyModel = CanonicalKeys.TokenizerModel;

    /// <summary>Canonical metadata key of <see cref="PreTokenizer"/>.</summary>
    public const string KeyPre = CanonicalKeys.TokenizerPre;

    /// <summary>Canonical metadata key of <see cref="Tokens"/>.</summary>
    public const string KeyTokens = CanonicalKeys.TokenizerTokens;

    /// <summary>Canonical metadata key of <see cref="TokenTypes"/>.</summary>
    public const string KeyTokenType = CanonicalKeys.TokenizerTokenTypes;

    /// <summary>Canonical metadata key of <see cref="Merges"/>.</summary>
    public const string KeyMerges = CanonicalKeys.TokenizerMerges;

    /// <summary>Reads the canonical tokenizer keys (<see cref="CanonicalKeys"/>). Lenient about numeric widths and array element types.</summary>
    /// <param name="md">Canonical model metadata.</param>
    /// <returns>The tokenizer description; the model defaults to "gpt2" when the metadata names none.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="md"/> is null.</exception>
    /// <exception cref="InvalidDataException">The token list is missing or a numeric value cannot be interpreted.</exception>
    public static TokenizerData FromMetadata(IReadOnlyDictionary<string, object> md)
    {
        ArgumentNullException.ThrowIfNull(md);
        string model = TokenizerMetadataReader.GetString(md, KeyModel) ?? "gpt2";
        string[] tokens = TokenizerMetadataReader.GetStringArray(md, KeyTokens)
            ?? throw new InvalidDataException($"Metadata key '{KeyTokens}' is missing or not a string array.");

        int? eos = TokenizerMetadataReader.GetInt(md, CanonicalKeys.TokenizerEosId);
        int? eot = TokenizerMetadataReader.GetInt(md, CanonicalKeys.TokenizerEotId);
        int? eom = TokenizerMetadataReader.GetInt(md, CanonicalKeys.TokenizerEomId);
        var eosIds = new List<int>(3);
        foreach (int? id in new[] { eos, eot, eom })
        {
            if (id is int v && !eosIds.Contains(v))
            {
                eosIds.Add(v);
            }
        }

        return new TokenizerData
        {
            Model = model,
            PreTokenizer = TokenizerMetadataReader.GetString(md, KeyPre),
            Tokens = tokens,
            TokenTypes = TokenizerMetadataReader.GetIntArray(md, KeyTokenType),
            Merges = TokenizerMetadataReader.GetStringArray(md, KeyMerges),
            Scores = TokenizerMetadataReader.GetFloatArray(md, CanonicalKeys.TokenizerScores),
            AddSpacePrefix = TokenizerMetadataReader.GetBool(md, CanonicalKeys.TokenizerAddSpacePrefix),
            Lowercase = TokenizerMetadataReader.GetBool(md, CanonicalKeys.TokenizerLowercase),
            StripAccents = TokenizerMetadataReader.GetBool(md, CanonicalKeys.TokenizerStripAccents),
            BosId = TokenizerMetadataReader.GetInt(md, CanonicalKeys.TokenizerBosId),
            EosId = eos,
            EosIds = eosIds.ToArray(),
            PadId = TokenizerMetadataReader.GetInt(md, CanonicalKeys.TokenizerPadId),
            UnkId = TokenizerMetadataReader.GetInt(md, CanonicalKeys.TokenizerUnkId),
            AddBos = TokenizerMetadataReader.GetBool(md, CanonicalKeys.TokenizerAddBos),
            AddEos = TokenizerMetadataReader.GetBool(md, CanonicalKeys.TokenizerAddEos),
        };
    }
}
