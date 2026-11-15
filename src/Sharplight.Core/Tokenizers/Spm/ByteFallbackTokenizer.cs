using System.Globalization;
using System.Text;

namespace Sharplight.Tokenizers.Spm;

/// <summary>
/// Shared vocabulary handling of the ▁-based tokenizers with byte fallback (SentencePiece "spm" and ranked "bpe"): token types,
/// <c>&lt;0xXX&gt;</c> byte tokens, special-token splitting, end-of-generation ids and detokenization.
/// </summary>
internal abstract class ByteFallbackTokenizer : ITokenizer
{
    private readonly byte[][] _bytes;   // UTF-8 of each vocabulary string
    private int[] _endOfGeneration = [];

    /// <summary>Creates the shared state; derived classes build their lookup structures and then call <see cref="ResolveEndOfGeneration"/>.</summary>
    /// <param name="data">The tokenizer description.</param>
    /// <param name="defaultBosId">BOS id used when the data names none.</param>
    /// <param name="defaultEosId">EOS id used when the data names none.</param>
    /// <param name="defaultAddSpacePrefix">Dummy-prefix default when the data does not say.</param>
    protected ByteFallbackTokenizer(TokenizerData data, int? defaultBosId, int? defaultEosId, bool defaultAddSpacePrefix)
    {
        ArgumentNullException.ThrowIfNull(data);
        int vocab = data.Tokens.Length;
        Tokens = data.Tokens;
        Types = TokenizerSetup.ResolveTypes(data);
        _bytes = new byte[vocab][];
        Array.Fill(ByteToken, -1);
        for (int i = 0; i < vocab; i++)
        {
            _bytes[i] = Encoding.UTF8.GetBytes(Tokens[i]);
            if (Types[i] == TokenType.Byte && TryParseByteToken(Tokens[i], out byte b))
            {
                ByteToken[b] = i;
            }
        }

        BosId = TokenizerSetup.ValidId(data.BosId ?? defaultBosId, vocab);
        EosId = TokenizerSetup.ValidId(data.EosId ?? defaultEosId, vocab);
        PadId = TokenizerSetup.ValidId(data.PadId, vocab);
        AddBos = data.AddBos ?? true;
        AddEos = data.AddEos ?? false;
        AddSpacePrefix = data.AddSpacePrefix ?? defaultAddSpacePrefix;
        Specials = new SpecialTokens(Tokens, i => Types[i], orderByUtf8Bytes: true);
    }

    /// <summary>The vocabulary strings.</summary>
    protected string[] Tokens { get; }

    /// <summary>Type of every token.</summary>
    protected TokenType[] Types { get; }

    /// <summary>Token id of each byte value's <c>&lt;0xXX&gt;</c> token, or -1 when the vocabulary lacks it.</summary>
    protected int[] ByteToken { get; } = new int[256];

    /// <summary>Splitter for special tokens occurring in text.</summary>
    protected SpecialTokens Specials { get; }

    /// <summary>Whether a dummy space is prepended to text (and removed again when decoding).</summary>
    protected bool AddSpacePrefix { get; }

    /// <inheritdoc/>
    public int VocabSize => Tokens.Length;

    /// <inheritdoc/>
    public int? BosId { get; }

    /// <inheritdoc/>
    public int? EosId { get; }

    /// <inheritdoc/>
    public int? PadId { get; }

    /// <inheritdoc/>
    public bool AddBos { get; }

    /// <inheritdoc/>
    public bool AddEos { get; }

    /// <inheritdoc/>
    public IReadOnlyList<int> EndOfGenerationIds => _endOfGeneration;

    /// <summary>Computes the end-of-generation ids; call once the vocabulary lookup behind <paramref name="lookup"/> is ready.</summary>
    protected void ResolveEndOfGeneration(TokenizerData data, string[] knownNames, Func<string, int?> lookup)
    {
        _endOfGeneration = TokenizerSetup.ResolveEndOfGeneration(data, EosId, Types, knownNames, lookup);
    }

    /// <summary>Parses a byte-fallback token string of the form <c>&lt;0xXX&gt;</c>.</summary>
    protected static bool TryParseByteToken(string token, out byte value)
    {
        value = 0;
        return token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token[5] == '>'
            && byte.TryParse(token.AsSpan(3, 2), NumberStyles.HexNumber, null, out value);
    }

    /// <inheritdoc/>
    public string? TokenString(int id) => (uint)id < (uint)Tokens.Length ? Tokens[id] : null;

    /// <inheritdoc/>
    public TokenType GetTokenType(int id) => (uint)id < (uint)Types.Length ? Types[id] : TokenType.Undefined;

    /// <inheritdoc/>
    public bool IsSpecial(int id) => GetTokenType(id) is TokenType.Control or TokenType.Unknown;

    /// <inheritdoc/>
    public bool IsEndOfGeneration(int id) => Array.BinarySearch(_endOfGeneration, id) >= 0;

    /// <inheritdoc/>
    public abstract int? TokenId(string piece);

    /// <inheritdoc/>
    public abstract int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true);

    /// <summary>Bytes a token renders to: normal pieces with ▁ turned into spaces, byte tokens as the byte, control tokens literally.</summary>
    public byte[] TokenBytes(int id)
    {
        if ((uint)id >= (uint)Tokens.Length)
        {
            return [];
        }

        switch (Types[id])
        {
            case TokenType.Byte:
                return TryParseByteToken(Tokens[id], out byte b) ? [b] : [];
            case TokenType.Unused:
                return [];
            case TokenType.Normal:
                return Encoding.UTF8.GetBytes(Tokens[id].Replace('▁', ' '));
            default:
                return _bytes[id];
        }
    }

    /// <inheritdoc/>
    public byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false)
    {
        var ms = new MemoryStream();

        // llama.cpp: the dummy prefix space is removed from the first piece; a skipped leading BOS keeps it.
        bool removeSpace = AddSpacePrefix;
        if (skipSpecial && AddBos && ids.Length > 0 && ids[0] == BosId)
        {
            removeSpace = false;
            ids = ids[1..];
        }

        foreach (int id in ids)
        {
            if (skipSpecial && IsSpecial(id))
            {
                continue;
            }

            var piece = TokenBytes(id);
            if (removeSpace && piece.Length > 0 && piece[0] == (byte)' ')
            {
                piece = piece[1..];
            }

            removeSpace = false;
            ms.Write(piece);
        }

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false) => Encoding.UTF8.GetString(DecodeBytes(ids, skipSpecial));
}
