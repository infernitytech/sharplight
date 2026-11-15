using System.Text;

namespace Sharplight.Tokenizers.Rwkv;

/// <summary>
/// The RWKV "World" tokenizer: a byte-level vocabulary applied by greedy longest match (a byte trie), no pre-tokenization, no BOS.
/// Vocabulary strings use Python-style escapes (<c>\xNN</c>, <c>\n</c>, <c>\t</c>, <c>\r</c>, <c>\\</c>) for bytes that are not printable; a position no token matches
/// yields the unknown token.
/// </summary>
internal sealed class RwkvWorldTokenizer : ITokenizer
{
    private readonly string[] _tokens;
    private readonly byte[][] _bytes;
    private readonly TokenType[] _types;
    private readonly Dictionary<long, int> _children = [];   // (node << 8 | byte) -> node
    private readonly List<int> _terminal = [-1];             // token id ending at each node, or -1
    private readonly Dictionary<string, int> _byString = [];
    private readonly int[] _endOfGeneration;
    private readonly int _unknownId;

    /// <summary>Builds the trie of the vocabulary.</summary>
    public RwkvWorldTokenizer(TokenizerData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        _tokens = data.Tokens;
        _types = TokenizerSetup.ResolveTypes(data);
        _bytes = new byte[_tokens.Length][];
        for (int id = 0; id < _tokens.Length; id++)
        {
            _bytes[id] = Unescape(_tokens[id]);
            _byString.TryAdd(_tokens[id], id);
            if (_types[id] is TokenType.Normal or TokenType.UserDefined && _bytes[id].Length > 0)
            {
                Insert(_bytes[id], id);
            }
        }

        BosId = TokenizerSetup.ValidId(data.BosId, _tokens.Length);
        EosId = TokenizerSetup.ValidId(data.EosId, _tokens.Length);
        PadId = TokenizerSetup.ValidId(data.PadId, _tokens.Length);
        _unknownId = TokenizerSetup.ValidId(data.UnkId, _tokens.Length) ?? 0;
        _endOfGeneration = TokenizerSetup.ResolveEndOfGeneration(data, EosId, _types, [], _ => null);
    }

    /// <inheritdoc/>
    public int VocabSize => _tokens.Length;

    /// <inheritdoc/>
    public int? BosId { get; }

    /// <inheritdoc/>
    public int? EosId { get; }

    /// <inheritdoc/>
    public int? PadId { get; }

    /// <inheritdoc/>
    public bool AddBos => false;

    /// <inheritdoc/>
    public bool AddEos => false;

    /// <inheritdoc/>
    public IReadOnlyList<int> EndOfGenerationIds => _endOfGeneration;

    private void Insert(byte[] bytes, int id)
    {
        int node = 0;
        foreach (byte b in bytes)
        {
            long key = ((long)node << 8) | b;
            if (!_children.TryGetValue(key, out int child))
            {
                child = _terminal.Count;
                _terminal.Add(-1);
                _children[key] = child;
            }

            node = child;
        }

        _terminal[node] = id;
    }

    private static byte[] Unescape(string text)
    {
        var bytes = new List<byte>(text.Length);
        var utf8 = new byte[4];
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c == '\\' && i + 1 < text.Length)
            {
                char next = text[i + 1];
                if (next == 'x' && i + 3 < text.Length && byte.TryParse(text.AsSpan(i + 2, 2), System.Globalization.NumberStyles.HexNumber, null, out byte value))
                {
                    bytes.Add(value);
                    i += 3;
                    continue;
                }

                byte? simple = next switch { 'n' => (byte)'\n', 't' => (byte)'\t', 'r' => (byte)'\r', '\\' => (byte)'\\', '\'' => (byte)'\'', '"' => (byte)'"', _ => null };
                if (simple is byte s)
                {
                    bytes.Add(s);
                    i++;
                    continue;
                }
            }

            int length = char.IsHighSurrogate(c) && i + 1 < text.Length ? 2 : 1;
            int written = Encoding.UTF8.GetBytes(text.AsSpan(i, length), utf8);
            bytes.AddRange(utf8.AsSpan(0, written));
            i += length - 1;
        }

        return [.. bytes];
    }

    /// <inheritdoc/>
    public int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        var input = Encoding.UTF8.GetBytes(text);
        var ids = new List<int>(input.Length / 2 + 1);
        int position = 0;
        while (position < input.Length)
        {
            int node = 0;
            int best = -1;
            int bestEnd = position;
            for (int i = position; i < input.Length && _children.TryGetValue(((long)node << 8) | input[i], out int child); i++)
            {
                node = child;
                if (_terminal[node] >= 0)
                {
                    best = _terminal[node];
                    bestEnd = i + 1;
                }
            }

            if (best < 0)
            {
                ids.Add(_unknownId);
                position++;
            }
            else
            {
                ids.Add(best);
                position = bestEnd;
            }
        }

        return [.. ids];
    }

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false) => Encoding.UTF8.GetString(DecodeBytes(ids, skipSpecial));

    /// <inheritdoc/>
    public byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false)
    {
        var bytes = new List<byte>();
        foreach (int id in ids)
        {
            if ((uint)id < (uint)_bytes.Length && !(skipSpecial && IsSpecial(id)))
            {
                bytes.AddRange(_bytes[id]);
            }
        }

        return [.. bytes];
    }

    /// <inheritdoc/>
    public byte[] TokenBytes(int id) => (uint)id < (uint)_bytes.Length ? _bytes[id] : [];

    /// <inheritdoc/>
    public string? TokenString(int id) => (uint)id < (uint)_tokens.Length ? _tokens[id] : null;

    /// <inheritdoc/>
    public TokenType GetTokenType(int id) => (uint)id < (uint)_types.Length ? _types[id] : TokenType.Undefined;

    /// <inheritdoc/>
    public bool IsSpecial(int id) => GetTokenType(id) is TokenType.Control or TokenType.Unknown;

    /// <inheritdoc/>
    public bool IsEndOfGeneration(int id) => Array.BinarySearch(_endOfGeneration, id) >= 0;

    /// <inheritdoc/>
    public int? TokenId(string piece) => _byString.TryGetValue(piece, out int id) ? id : null;
}
