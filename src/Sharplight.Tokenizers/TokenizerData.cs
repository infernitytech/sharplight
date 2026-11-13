using Sharplight.Models;
using System.Collections;
using System.Globalization;

namespace Sharplight.Tokenizers;

/// <summary>Token classification (canonical <c>tokenizer.token_type</c> values).</summary>
public enum TokenType
{
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

/// <summary>
/// Format-agnostic description of a tokenizer. Build it by hand or via
/// <see cref="FromMetadata"/> from canonical model metadata (<see cref="CanonicalKeys"/>).
/// </summary>
public sealed record TokenizerData
{
    /// <summary>
    /// Tokenizer algorithm: "gpt2" (byte-level BPE), "spm" (SentencePiece-style: scored merges, ▁ for spaces, byte fallback),
    /// "wordpiece" (BERT: greedy longest match, ## continuations) or "unigram" (SentencePiece unigram: Viterbi over piece scores).
    /// </summary>
    public required string Model { get; init; }

    /// <summary>Pre-tokenizer preset name, e.g. "qwen35", "qwen2", "llama-bpe", "gpt-2", "default". Null/empty means "default".</summary>
    public string? PreTokenizer { get; init; }

    public required string[] Tokens { get; init; }

    /// <summary>Per-token <see cref="TokenType"/> as ints. Null means all tokens are normal.</summary>
    public int[]? TokenTypes { get; init; }

    /// <summary>BPE merges, "left right" per entry, in rank order.</summary>
    public string[]? Merges { get; init; }

    /// <summary>Per-token scores of a SentencePiece tokenizer ("spm").</summary>
    public float[]? Scores { get; init; }

    /// <summary>SentencePiece: prepend a space to the text (default true).</summary>
    public bool? AddSpacePrefix { get; init; }

    /// <summary>WordPiece: lowercase the text (default true).</summary>
    public bool? Lowercase { get; init; }

    /// <summary>WordPiece: strip accents (default: same as <see cref="Lowercase"/>).</summary>
    public bool? StripAccents { get; init; }

    public int? BosId { get; init; }

    /// <summary>The primary EOS id (used when <see cref="AddEos"/> is on).</summary>
    public int? EosId { get; init; }

    /// <summary>All ids that the metadata declares as end-of-generation (eos, eot, eom). Additional EOG tokens are detected by name.</summary>
    public int[] EosIds { get; init; } = [];

    public int? PadId { get; init; }
    public int? UnkId { get; init; }

    /// <summary>Whether to prepend BOS. Null = use the pre-tokenizer preset's default.</summary>
    public bool? AddBos { get; init; }

    /// <summary>Whether to append EOS. Null = false.</summary>
    public bool? AddEos { get; init; }

    public const string KeyModel = CanonicalKeys.TokenizerModel;
    public const string KeyPre = CanonicalKeys.TokenizerPre;
    public const string KeyTokens = CanonicalKeys.TokenizerTokens;
    public const string KeyTokenType = CanonicalKeys.TokenizerTokenTypes;
    public const string KeyMerges = CanonicalKeys.TokenizerMerges;

    /// <summary>Reads the canonical tokenizer keys (<see cref="CanonicalKeys"/>). Lenient about numeric widths and array element types.</summary>
    public static TokenizerData FromMetadata(IReadOnlyDictionary<string, object> md)
    {
        ArgumentNullException.ThrowIfNull(md);
        string model = GetString(md, KeyModel) ?? "gpt2";
        string[] tokens = GetStringArray(md, KeyTokens)
            ?? throw new InvalidDataException($"Metadata key '{KeyTokens}' is missing or not a string array.");

        int? eos = GetInt(md, CanonicalKeys.TokenizerEosId);
        int? eot = GetInt(md, CanonicalKeys.TokenizerEotId);
        int? eom = GetInt(md, CanonicalKeys.TokenizerEomId);
        var eosIds = new List<int>(3);
        foreach (int? id in new[] { eos, eot, eom })
            if (id is int v && !eosIds.Contains(v)) eosIds.Add(v);

        return new TokenizerData
        {
            Model = model,
            PreTokenizer = GetString(md, KeyPre),
            Tokens = tokens,
            TokenTypes = GetIntArray(md, KeyTokenType),
            Merges = GetStringArray(md, KeyMerges),
            Scores = GetFloatArray(md, CanonicalKeys.TokenizerScores),
            AddSpacePrefix = GetBool(md, CanonicalKeys.TokenizerAddSpacePrefix),
            Lowercase = GetBool(md, CanonicalKeys.TokenizerLowercase),
            StripAccents = GetBool(md, CanonicalKeys.TokenizerStripAccents),
            BosId = GetInt(md, CanonicalKeys.TokenizerBosId),
            EosId = eos,
            EosIds = eosIds.ToArray(),
            PadId = GetInt(md, CanonicalKeys.TokenizerPadId),
            UnkId = GetInt(md, CanonicalKeys.TokenizerUnkId),
            AddBos = GetBool(md, CanonicalKeys.TokenizerAddBos),
            AddEos = GetBool(md, CanonicalKeys.TokenizerAddEos),
        };
    }

    private static string? GetString(IReadOnlyDictionary<string, object> md, string key)
        => md.TryGetValue(key, out var v) ? v as string ?? v?.ToString() : null;

    private static int? GetInt(IReadOnlyDictionary<string, object> md, string key)
    {
        if (!md.TryGetValue(key, out var v) || v is null) return null;
        long n = ToLong(v);
        // llama.cpp treats ids >= vocab as invalid; 0xFFFFFFFF style values mean "none".
        return n is < 0 or > int.MaxValue ? null : (int)n;
    }

    private static bool? GetBool(IReadOnlyDictionary<string, object> md, string key)
    {
        if (!md.TryGetValue(key, out var v) || v is null) return null;
        return v is bool b ? b : ToLong(v) != 0;
    }

    private static string[]? GetStringArray(IReadOnlyDictionary<string, object> md, string key)
    {
        if (!md.TryGetValue(key, out var v) || v is null) return null;
        if (v is string[] sa) return sa;
        if (v is string) return null;
        if (v is not IEnumerable e) return null;
        var list = new List<string>();
        foreach (object? o in e) list.Add(o as string ?? o?.ToString() ?? "");
        return list.ToArray();
    }

    private static float[]? GetFloatArray(IReadOnlyDictionary<string, object> md, string key)
    {
        if (!md.TryGetValue(key, out var v) || v is null) return null;
        if (v is float[] fa) return fa;
        if (v is string || v is not IEnumerable e) return null;
        var list = new List<float>();
        foreach (object? o in e) list.Add(Convert.ToSingle(o, CultureInfo.InvariantCulture));
        return list.ToArray();
    }

    private static int[]? GetIntArray(IReadOnlyDictionary<string, object> md, string key)
    {
        if (!md.TryGetValue(key, out var v) || v is null) return null;
        if (v is int[] ia) return ia;
        if (v is string || v is not IEnumerable e) return null;
        var list = new List<int>();
        foreach (object? o in e) list.Add(checked((int)ToLong(o!)));
        return list.ToArray();
    }

    private static long ToLong(object v) => v switch
    {
        long l => l,
        int i => i,
        uint u => u,
        ulong ul => ul > long.MaxValue ? -1 : (long)ul,
        short s => s,
        ushort us => us,
        byte b => b,
        sbyte sb => sb,
        bool bo => bo ? 1 : 0,
        float f => (long)f,
        double d => (long)d,
        IConvertible c => c.ToInt64(CultureInfo.InvariantCulture),
        _ => throw new InvalidDataException($"Cannot interpret {v.GetType()} as a number."),
    };
}
