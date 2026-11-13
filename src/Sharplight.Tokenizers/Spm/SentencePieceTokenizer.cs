using System.Text;

namespace Sharplight.Tokenizers.Spm;

/// <summary>
/// SentencePiece-style tokenizer ("spm"; llama.cpp's <c>LLAMA_VOCAB_TYPE_SPM</c>): text is split into UTF-8 characters
/// (spaces become ▁), adjacent pairs whose concatenation is a vocabulary token are merged best-score-first, symbols
/// that never became tokens fall back to byte tokens <c>&lt;0xXX&gt;</c>. Used by Llama 2, Mistral, Gemma, Phi-3.
/// </summary>
public sealed class SentencePieceTokenizer : ITokenizer
{
    private static readonly string[] EogNames =
    [
        "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "</s>", "<eos>", "<|end_of_text|>", "[EOS]",
    ];

    private static readonly byte[] Space = "▁"u8.ToArray();   // ▁ (U+2581) in UTF-8

    private readonly string[] _tokens;
    private readonly byte[][] _bytes;          // UTF-8 of each vocabulary string
    private readonly float[] _scores;
    private readonly TokenType[] _types;
    private readonly Dictionary<string, int> _toId = new(StringComparer.Ordinal);   // Latin-1 view of the UTF-8 bytes
    private readonly int[] _byteToken = new int[256];
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

    public SentencePieceTokenizer(TokenizerData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        int n = data.Tokens.Length;
        _tokens = data.Tokens;
        _bytes = new byte[n][];
        _scores = data.Scores is { } sc && sc.Length >= n ? sc : new float[n];
        _types = new TokenType[n];
        Array.Fill(_byteToken, -1);
        for (int i = 0; i < n; i++)
        {
            _types[i] = data.TokenTypes is { } tt && i < tt.Length && Enum.IsDefined((TokenType)tt[i]) ? (TokenType)tt[i] : TokenType.Normal;
            _bytes[i] = Encoding.UTF8.GetBytes(_tokens[i]);
            _toId[Latin1(_bytes[i])] = i;
            if (_types[i] == TokenType.Byte && TryParseByteToken(_tokens[i], out byte b))
            {
                _byteToken[b] = i;
            }
        }
        int? Valid(int? id) => id is int v && v >= 0 && v < n ? v : null;
        BosId = Valid(data.BosId ?? 1);
        EosId = Valid(data.EosId ?? 2);
        PadId = Valid(data.PadId);
        AddBos = data.AddBos ?? true;
        AddEos = data.AddEos ?? false;
        _addSpacePrefix = data.AddSpacePrefix ?? true;

        _specials = Enumerable.Range(0, n).Where(i => _types[i] is TokenType.Control or TokenType.UserDefined or TokenType.Unknown)
            .Select(i => (_tokens[i], i)).OrderByDescending(s => _bytes[s.i].Length).ToArray();

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
            if (_toId.TryGetValue(Latin1(Encoding.UTF8.GetBytes(name)), out int id) && _types[id] is TokenType.Control or TokenType.UserDefined && !eog.Contains(id))
            {
                eog.Add(id);
            }
        }

        eog.Sort();
        _eog = [.. eog];
    }

    private static string Latin1(ReadOnlySpan<byte> b) => Encoding.Latin1.GetString(b);

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
    public int? TokenId(string piece) => _toId.TryGetValue(Latin1(Encoding.UTF8.GetBytes(piece)), out int id) ? id : null;

    /// <summary>Bytes a token renders to: normal pieces with ▁ turned into spaces, byte tokens as the byte, control tokens literally.</summary>
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
            case TokenType.Normal:
                return Encoding.UTF8.GetBytes(_tokens[id].Replace('▁', ' '));
            default: return _bytes[id];
        }
    }

    public byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false)
    {
        var ms = new MemoryStream();
        // llama.cpp: the dummy prefix space is removed from the first piece; a skipped leading BOS keeps it.
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

        bool prevSpecial = true;   // prefix with a space if first
        foreach (var (isToken, piece, id) in Partition(text, parseSpecial))
        {
            if (isToken) { output.Add(id); prevSpecial = true; continue; }
            string t = (_addSpacePrefix && prevSpecial ? " " : "") + piece;
            EncodeText(Encoding.UTF8.GetBytes(t.Replace(" ", "▁")), output);
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

            if (stext.Length == 0)
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

    private struct Symbol { public int Start, Len, Prev, Next; }
    private readonly record struct Bigram(int Left, int Right, float Score, int Size);

    /// <summary>Port of llama.cpp's <c>llm_tokenizer_spm_session::tokenize</c>.</summary>
    private void EncodeText(byte[] text, List<int> output)
    {
        if (text.Length == 0)
        {
            return;
        }

        var symbols = new List<Symbol>(text.Length);
        for (int offs = 0; offs < text.Length;)
        {
            int len = Math.Min(Utf8Len(text[offs]), text.Length - offs);
            symbols.Add(new Symbol { Start = offs, Len = len, Prev = symbols.Count - 1, Next = offs + len == text.Length ? -1 : symbols.Count + 1 });
            offs += len;
        }
        var queue = new PriorityQueue<Bigram, (float, int)>();   // min-heap on (-score, left)
        var revMerge = new Dictionary<string, (int Left, int Right)>(StringComparer.Ordinal);

        void TryAdd(int left, int right)
        {
            if (left == -1 || right == -1)
            {
                return;
            }

            int size = symbols[left].Len + symbols[right].Len;
            string key = Latin1(text.AsSpan(symbols[left].Start, size));
            if (!_toId.TryGetValue(key, out int id))
            {
                return;
            }

            float score = _scores[id];
            queue.Enqueue(new Bigram(left, right, score, size), (-score, left));
            revMerge[key] = (left, right);
        }

        for (int i = 1; i < symbols.Count; i++)
        {
            TryAdd(i - 1, i);
        }

        while (queue.Count > 0)
        {
            var bg = queue.Dequeue();
            var l = symbols[bg.Left]; var r = symbols[bg.Right];
            if (l.Len == 0 || r.Len == 0 || l.Len + r.Len != bg.Size)
            {
                continue;
            }

            l.Len += r.Len; r.Len = 0;
            l.Next = r.Next;
            symbols[bg.Left] = l; symbols[bg.Right] = r;
            if (r.Next >= 0) { var nx = symbols[r.Next]; nx.Prev = bg.Left; symbols[r.Next] = nx; }
            TryAdd(l.Prev, bg.Left);
            TryAdd(bg.Left, l.Next);
        }

        var todo = new Stack<int>();
        for (int i = 0; i != -1; i = symbols[i].Next)
        {
            todo.Push(i);
            while (todo.Count > 0)
            {
                var cur = symbols[todo.Pop()];
                string key = Latin1(text.AsSpan(cur.Start, cur.Len));
                if (_toId.TryGetValue(key, out int tok)) { output.Add(tok); continue; }
                if (!revMerge.TryGetValue(key, out var lr))
                {
                    for (int j = 0; j < cur.Len; j++)
                    {
                        if (_byteToken[text[cur.Start + j]] is var bt and >= 0)
                        {
                            output.Add(bt);
                        }
                    }

                    continue;
                }
                todo.Push(lr.Right);
                todo.Push(lr.Left);
            }
        }
    }

    private static int Utf8Len(byte b) => b < 0xC0 ? 1 : b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
}
