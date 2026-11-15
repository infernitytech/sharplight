using System.Collections.Concurrent;
using System.Text;
using System.Text.RegularExpressions;
using Sharplight.Tokenizers.Bpe;

namespace Sharplight.Tokenizers.Clip;

/// <summary>
/// The byte-level BPE tokenizer of CLIP text encoders ("clip"; OpenAI CLIP, OpenCLIP, the text encoders of Stable Diffusion):
/// whitespace runs collapse to one space and the text is lowercased; it splits into contractions, letter runs, single digits
/// and runs of other symbols; each piece is byte-mapped like GPT-2 and its last symbol carries the end-of-word marker
/// <c>&lt;/w&gt;</c> before the merges apply. Vocabulary strings are in that form (<c>cat&lt;/w&gt;</c>).
/// </summary>
internal sealed partial class ClipBpeTokenizer : ITokenizer
{
    private const string EndOfWord = "</w>";
    private const int MaxCachedWords = 1 << 16;

    private readonly string[] _tokens;
    private readonly TokenType[] _types;
    private readonly Dictionary<string, int> _toId = new(StringComparer.Ordinal);
    private readonly Dictionary<(string Left, string Right), int> _ranks = [];
    private readonly ConcurrentDictionary<string, int[]> _cache = new(StringComparer.Ordinal);
    private readonly SpecialTokens _specials;
    private readonly int? _unk;

    /// <inheritdoc/>
    public int VocabSize => _tokens.Length;
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
    public IReadOnlyList<int> EndOfGenerationIds { get; }

    /// <summary>Creates the tokenizer from a "clip" description (vocabulary with <c>&lt;/w&gt;</c> markers and merges "a b").</summary>
    /// <exception cref="InvalidDataException">The description has no merges.</exception>
    public ClipBpeTokenizer(TokenizerData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        _tokens = data.Tokens;
        _types = TokenizerSetup.ResolveTypes(data);
        for (int i = 0; i < _tokens.Length; i++)
        {
            _toId.TryAdd(_tokens[i], i);
        }

        var merges = data.Merges ?? throw new InvalidDataException("A CLIP tokenizer needs merges.");
        for (int rank = 0; rank < merges.Length; rank++)
        {
            int space = merges[rank].IndexOf(' ', 1);
            if (space > 0)
            {
                _ranks.TryAdd((merges[rank][..space], merges[rank][(space + 1)..]), rank);
            }
        }

        int? Valid(int? id) => TokenizerSetup.ValidId(id, _tokens.Length);
        int? Named(string s) => _toId.TryGetValue(s, out int id) ? id : null;
        BosId = Valid(data.BosId) ?? Named("<|startoftext|>");
        EosId = Valid(data.EosId) ?? Named("<|endoftext|>");
        PadId = Valid(data.PadId) ?? EosId;
        _unk = Valid(data.UnkId) ?? EosId;
        foreach (int? id in new[] { BosId, EosId })
        {
            if (id is int special && _types[special] == TokenType.Normal)
            {
                _types[special] = TokenType.Control;
            }
        }

        AddBos = data.AddBos ?? true;
        AddEos = data.AddEos ?? true;
        EndOfGenerationIds = EosId is int eos ? [eos] : [];
        _specials = new SpecialTokens(_tokens, i => _types[i]);
    }

    /// <inheritdoc/>
    public string? TokenString(int id) => (uint)id < (uint)_tokens.Length ? _tokens[id] : null;
    /// <inheritdoc/>
    public TokenType GetTokenType(int id) => (uint)id < (uint)_types.Length ? _types[id] : TokenType.Undefined;
    /// <inheritdoc/>
    public bool IsSpecial(int id) => GetTokenType(id) is TokenType.Control or TokenType.Unknown;
    /// <inheritdoc/>
    public bool IsEndOfGeneration(int id) => EosId == id;
    /// <inheritdoc/>
    public int? TokenId(string piece) => _toId.TryGetValue(piece, out int id) ? id : null;

    /// <inheritdoc/>
    public byte[] TokenBytes(int id)
    {
        if ((uint)id >= (uint)_tokens.Length)
        {
            return [];
        }

        return _types[id] == TokenType.Normal ? PieceBytes(_tokens[id]) : Encoding.UTF8.GetBytes(_tokens[id]);
    }

    /// <inheritdoc/>
    public byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false)
    {
        var bytes = new List<byte>();
        foreach (int id in ids)
        {
            if ((uint)id >= (uint)_tokens.Length || skipSpecial && IsSpecial(id))
            {
                continue;
            }

            bytes.AddRange(TokenBytes(id));
        }

        return [.. bytes];
    }

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false) => Encoding.UTF8.GetString(DecodeBytes(ids, skipSpecial)).TrimEnd();

    /// <inheritdoc/>
    public int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        var output = new List<int>(text.Length / 3 + 2);
        if (addSpecial && AddBos && BosId is int bos)
        {
            output.Add(bos);
        }

        foreach (var (fragment, id) in _specials.Split(text, parseSpecial))
        {
            if (id >= 0)
            {
                output.Add(id);
                continue;
            }

            string clean = WhitespaceRuns().Replace(fragment, " ").Trim().ToLowerInvariant();
            foreach (Match piece in Pieces().Matches(clean))
            {
                output.AddRange(EncodePiece(piece.Value));
            }
        }

        if (addSpecial && AddEos && EosId is int eos)
        {
            output.Add(eos);
        }

        return [.. output];
    }

    /// <summary>The bytes of a vocabulary string: byte-mapped characters, with the end-of-word marker as a trailing space.</summary>
    private static byte[] PieceBytes(string piece)
    {
        bool endsWord = piece.EndsWith(EndOfWord, StringComparison.Ordinal);
        var body = endsWord ? piece.AsSpan(0, piece.Length - EndOfWord.Length) : piece.AsSpan();
        var bytes = new List<byte>(body.Length + 1);
        foreach (char c in body)
        {
            if (ByteEncoding.TryGetByte(c, out byte b))
            {
                bytes.Add(b);
            }
        }

        if (endsWord)
        {
            bytes.Add((byte)' ');
        }

        return [.. bytes];
    }

    private int[] EncodePiece(string piece)
    {
        if (_cache.TryGetValue(piece, out var cached))
        {
            return cached;
        }

        var utf8 = Encoding.UTF8.GetBytes(piece);
        var symbols = new List<string>(utf8.Length);
        foreach (byte b in utf8)
        {
            symbols.Add(ByteEncoding.ByteToChar[b].ToString());
        }

        symbols[^1] += EndOfWord;
        Merge(symbols);
        var ids = new int[symbols.Count];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = _toId.TryGetValue(symbols[i], out int id) ? id : _unk ?? 0;
        }

        if (_cache.Count < MaxCachedWords)
        {
            _cache.TryAdd(piece, ids);
        }

        return ids;
    }

    /// <summary>Applies the lowest-ranked merge of adjacent symbols until none applies (pieces are short, so quadratic is fine).</summary>
    private void Merge(List<string> symbols)
    {
        while (symbols.Count > 1)
        {
            int best = -1;
            int bestRank = int.MaxValue;
            for (int i = 0; i + 1 < symbols.Count; i++)
            {
                if (_ranks.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    best = i;
                }
            }

            if (best < 0)
            {
                return;
            }

            string left = symbols[best];
            string right = symbols[best + 1];
            for (int i = 0; i + 1 < symbols.Count; i++)
            {
                if (symbols[i] == left && symbols[i + 1] == right)
                {
                    symbols[i] = left + right;
                    symbols.RemoveAt(i + 1);
                }
            }
        }
    }

    [GeneratedRegex(@"\s+")]
    private static partial Regex WhitespaceRuns();

    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+")]
    private static partial Regex Pieces();
}
