using System.Text;
using Sharplight.Tokenizers.Bpe;

namespace Sharplight.Tokenizers.Spm;

/// <summary>
/// Character-level BPE with ▁ for spaces and byte fallback (model "bpe"; Gemma 4): text is split into Unicode characters
/// (spaces become ▁), characters missing from the vocabulary fall back to byte tokens <c>&lt;0xXX&gt;</c>, then the adjacent pair
/// with the lowest merge rank is merged repeatedly (the Hugging Face BPE model with <c>byte_fallback</c>; no pre-tokenization
/// beyond special-token splitting, so a whole prompt is one BPE unit). Merges are <c>"left right"</c> strings of vocabulary tokens.
/// </summary>
public sealed class RankedBpeTokenizer : ITokenizer
{
    private static readonly string[] EogNames =
    [
        "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<turn|>", "<|endoftext|>", "</s>", "<eos>", "<|end_of_text|>", "[EOS]",
    ];

    private readonly string[] _tokens;
    private readonly byte[][] _bytes;
    private readonly TokenType[] _types;
    private readonly Dictionary<string, int> _toId = new(StringComparer.Ordinal);
    private readonly int[] _byteToken = new int[256];
    private readonly MergeTable _merges;
    private readonly (string Text, int Id)[] _specials;
    private readonly int[] _eog;
    private readonly bool _addSpacePrefix;

    public int VocabSize => _tokens.Length;
    public int? BosId { get; }
    public int? EosId { get; }
    public int? PadId { get; }
    public bool AddBos { get; }
    public bool AddEos { get; }
    public IReadOnlyList<int> EndOfGenerationIds => _eog;

    public RankedBpeTokenizer(TokenizerData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        int n = data.Tokens.Length;
        _tokens = data.Tokens;
        _bytes = new byte[n][];
        _types = new TokenType[n];
        Array.Fill(_byteToken, -1);
        for (int i = 0; i < n; i++)
        {
            _types[i] = data.TokenTypes is { } tt && i < tt.Length && Enum.IsDefined((TokenType)tt[i]) ? (TokenType)tt[i] : TokenType.Normal;
            _bytes[i] = Encoding.UTF8.GetBytes(_tokens[i]);
            _toId[_tokens[i]] = i;
            if (_types[i] == TokenType.Byte && TryParseByteToken(_tokens[i], out byte b))
            {
                _byteToken[b] = i;
            }
        }
        int? Valid(int? id) => id is int v && v >= 0 && v < n ? v : null;
        BosId = Valid(data.BosId);
        EosId = Valid(data.EosId);
        PadId = Valid(data.PadId);
        AddBos = data.AddBos ?? true;
        AddEos = data.AddEos ?? false;
        _addSpacePrefix = data.AddSpacePrefix ?? false;

        var merges = data.Merges ?? [];
        _merges = new MergeTable(merges.Length);
        for (int r = 0; r < merges.Length; r++)
        {
            string m = merges[r];
            int sep = m.Length > 1 ? m.IndexOf(' ', 1) : -1;
            if (sep < 0)
            {
                continue;
            }

            if (_toId.TryGetValue(m[..sep], out int l) && _toId.TryGetValue(m[(sep + 1)..], out int rt) && _toId.TryGetValue(m[..sep] + m[(sep + 1)..], out int res))
            {
                _merges.TryAdd(l, rt, r, res);
            }
        }

        _specials = Enumerable.Range(0, n).Where(i => _types[i] is TokenType.Control or TokenType.UserDefined or TokenType.Unknown)
            .Select(i => (_tokens[i], i)).Where(s => s.Item1.Length > 0).OrderByDescending(s => _bytes[s.i].Length).ToArray();

        var eog = new List<int>();
        foreach (int id in data.EosIds)
        {
            if (Valid(id) is int v && !eog.Contains(v))
            {
                eog.Add(v);
            }
        }

        if (EosId is int e && !eog.Contains(e))
        {
            eog.Add(e);
        }

        foreach (string name in EogNames)
        {
            if (_toId.TryGetValue(name, out int id) && _types[id] is TokenType.Control or TokenType.UserDefined && !eog.Contains(id))
            {
                eog.Add(id);
            }
        }

        eog.Sort();
        _eog = [.. eog];
    }

    private static bool TryParseByteToken(string t, out byte b)
    {
        b = 0;
        return t.Length == 6 && t.StartsWith("<0x", StringComparison.Ordinal) && t[5] == '>'
            && byte.TryParse(t.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out b);
    }

    // ------------------------------------------------------------------ decoding

    public string? TokenString(int id) => (uint)id < (uint)_tokens.Length ? _tokens[id] : null;
    public TokenType GetTokenType(int id) => (uint)id < (uint)_types.Length ? _types[id] : TokenType.Undefined;
    public bool IsSpecial(int id) => GetTokenType(id) is TokenType.Control or TokenType.Unknown;
    public bool IsEndOfGeneration(int id) => Array.BinarySearch(_eog, id) >= 0;
    public int? TokenId(string piece) => _toId.TryGetValue(piece, out int id) ? id : null;

    public byte[] TokenBytes(int id)
    {
        if ((uint)id >= (uint)_tokens.Length)
        {
            return [];
        }

        switch (_types[id])
        {
            case TokenType.Byte: return TryParseByteToken(_tokens[id], out byte b) ? [b] : [];
            case TokenType.Unused: return [];
            case TokenType.Normal: return Encoding.UTF8.GetBytes(_tokens[id].Replace('▁', ' '));
            default: return _bytes[id];
        }
    }

    public byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false)
    {
        var ms = new MemoryStream();
        bool removeSpace = _addSpacePrefix;
        if (skipSpecial && AddBos && ids.Length > 0 && ids[0] == BosId) { removeSpace = false; ids = ids[1..]; }
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

    public string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false) => Encoding.UTF8.GetString(DecodeBytes(ids, skipSpecial));

    // ------------------------------------------------------------------ encoding

    public int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        var output = new List<int>(Math.Max(16, text.Length / 3 + 4));
        if (addSpecial && AddBos && BosId is int bos)
        {
            output.Add(bos);
        }

        bool prevSpecial = true;
        foreach (var (isToken, piece, id) in Partition(text, parseSpecial))
        {
            if (isToken) { output.Add(id); prevSpecial = true; continue; }
            string t = (_addSpacePrefix && prevSpecial ? " " : "") + piece;
            EncodeText(t.Replace(' ', '▁'), output);
            prevSpecial = false;
        }
        if (addSpecial && AddEos && EosId is int eos)
        {
            output.Add(eos);
        }

        return [.. output];
    }

    private List<(bool IsToken, string Text, int Id)> Partition(string text, bool parseSpecial)
    {
        var buffer = new List<(bool, string, int)>();
        if (text.Length > 0)
        {
            buffer.Add((false, text, 0));
        }

        foreach (var (stext, sid) in _specials)
        {
            if (!parseSpecial && _types[sid] is TokenType.Control or TokenType.Unknown)
            {
                continue;
            }

            var next = new List<(bool, string, int)>(buffer.Count + 2);
            foreach (var frag in buffer)
            {
                if (frag.Item1 || !frag.Item2.Contains(stext, StringComparison.Ordinal)) { next.Add(frag); continue; }
                string rest = frag.Item2;
                while (true)
                {
                    int at = rest.IndexOf(stext, StringComparison.Ordinal);
                    if (at < 0) { if (rest.Length > 0) { next.Add((false, rest, 0)); } break; }
                    if (at > 0)
                    {
                        next.Add((false, rest[..at], 0));
                    }

                    next.Add((true, "", sid));
                    rest = rest[(at + stext.Length)..];
                }
            }
            buffer = next;
        }
        return buffer;
    }

    private readonly record struct Bigram(int Left, int Right, int LeftId, int RightId, int Result);

    private void EncodeText(string text, List<int> output)
    {
        if (text.Length == 0)
        {
            return;
        }
        // initial symbols: one per character (or one byte token per UTF-8 byte of an unknown character)
        var ids = new List<int>(text.Length);
        for (int i = 0; i < text.Length;)
        {
            int len = char.IsHighSurrogate(text[i]) && i + 1 < text.Length ? 2 : 1;
            string ch = text.Substring(i, len);
            if (_toId.TryGetValue(ch, out int id))
            {
                ids.Add(id);
            }
            else
            {
                foreach (byte b in Encoding.UTF8.GetBytes(ch))
                {
                    if (_byteToken[b] >= 0)
                    {
                        ids.Add(_byteToken[b]);
                    }
                }
            }

            i += len;
        }
        int n = ids.Count;
        if (n == 0)
        {
            return;
        }

        var sym = ids.ToArray();
        var prev = new int[n]; var next = new int[n];
        for (int i = 0; i < n; i++) { prev[i] = i - 1; next[i] = i + 1 < n ? i + 1 : -1; }
        var queue = new PriorityQueue<Bigram, (int, int)>();
        void TryAdd(int l, int r)
        {
            if (l < 0 || r < 0)
            {
                return;
            }

            if (_merges.TryGet(sym[l], sym[r], out int rank, out int res))
            {
                queue.Enqueue(new Bigram(l, r, sym[l], sym[r], res), (rank, l));
            }
        }
        for (int i = 1; i < n; i++)
        {
            TryAdd(i - 1, i);
        }

        while (queue.Count > 0)
        {
            var bg = queue.Dequeue();
            if (sym[bg.Left] != bg.LeftId || sym[bg.Right] != bg.RightId || sym[bg.Left] < 0 || sym[bg.Right] < 0 || next[bg.Left] != bg.Right)
            {
                continue;
            }

            sym[bg.Left] = bg.Result;
            sym[bg.Right] = -1;
            next[bg.Left] = next[bg.Right];
            if (next[bg.Right] >= 0)
            {
                prev[next[bg.Right]] = bg.Left;
            }

            TryAdd(prev[bg.Left], bg.Left);
            TryAdd(bg.Left, next[bg.Left]);
        }
        for (int i = 0; i != -1; i = next[i])
        {
            output.Add(sym[i]);
        }
    }
}
