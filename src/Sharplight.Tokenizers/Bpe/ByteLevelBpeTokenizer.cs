using System.Buffers;
using System.Collections.Concurrent;
using System.Text;

namespace Sharplight.Tokenizers.Bpe;

/// <summary>
/// GPT-2 style byte-level BPE (model "gpt2"), behaviour-compatible with llama.cpp's <c>llm_tokenizer_bpe</c>.
/// Thread-safe after construction.
/// </summary>
public sealed class ByteLevelBpeTokenizer : ITokenizer
{
    private const int StackWordLimit = 256;
    private const int MaxCachedWordBytes = 128;
    private const int MaxCacheEntries = 400_000;

    // Names llama.cpp treats as end-of-generation tokens.
    private static readonly string[] EogNames =
    [
        "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>",
        "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
        "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>",
    ];

    private readonly string[] _tokens;
    private readonly TokenType[] _types;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _tokenToIdSpan;
    private readonly byte[]?[] _tokenBytes;
    private readonly int[] _initialSymbol = new int[256]; // byte -> symbol id (may be a virtual id >= VocabSize)
    private readonly MergeTable _merges;
    private readonly PreTokenizer _splitter;
    private readonly bool _ignoreMerges;
    private readonly (string Text, int Id)[] _specials; // longest first
    private readonly int[] _eogIds;
    private readonly HashSet<int> _eogSet;
    private readonly ConcurrentDictionary<byte[], int[]> _wordCache;
    private readonly ConcurrentDictionary<byte[], int[]>.AlternateLookup<ReadOnlySpan<byte>> _wordCacheSpan;

    /// <summary>Words with more symbols than this use a priority queue instead of repeated scans. Exposed for tests.</summary>
    internal int HeapThreshold { get; init; } = 192;

    public int VocabSize => _tokens.Length;
    public int? BosId { get; }
    public int? EosId { get; }
    public int? PadId { get; }
    public bool AddBos { get; }
    public bool AddEos { get; }
    public IReadOnlyList<int> EndOfGenerationIds => _eogIds;

    public ByteLevelBpeTokenizer(TokenizerData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        var preset = PreTokenizerPreset.Resolve(data.PreTokenizer);
        _splitter = preset.Splitter;
        _ignoreMerges = preset.IgnoreMerges;

        int vocab = data.Tokens.Length;
        _tokens = new string[vocab];
        _types = new TokenType[vocab];
        _tokenBytes = new byte[vocab][];
        _tokenToId = new Dictionary<string, int>(vocab, StringComparer.Ordinal);
        for (int i = 0; i < vocab; i++)
        {
            string t = data.Tokens[i];
            if (t.Length == 0)
            {
                t = $"[EMPTY_{i}]";
            }

            _tokens[i] = t;
            _tokenToId[t] = i; // last duplicate wins, as in llama.cpp
            _types[i] = data.TokenTypes is { } tt && i < tt.Length && Enum.IsDefined((TokenType)tt[i]) ? (TokenType)tt[i] : TokenType.Normal;
        }
        _tokenToIdSpan = _tokenToId.GetAlternateLookup<ReadOnlySpan<char>>();

        int? ValidId(int? id) => id is int v && v >= 0 && v < vocab ? v : null;
        BosId = ValidId(data.BosId);
        EosId = ValidId(data.EosId);
        PadId = ValidId(data.PadId);
        AddBos = data.AddBos ?? preset.DefaultAddBos;
        AddEos = data.AddEos ?? false;

        _merges = BuildMerges(data.Merges ?? []);

        // special tokens, longest (UTF-8 bytes) first; stable for equal lengths
        var specials = new List<(string, int, int)>();
        for (int i = 0; i < vocab; i++)
        {
            if (_types[i] is TokenType.Control or TokenType.UserDefined or TokenType.Unknown)
            {
                specials.Add((_tokens[i], i, Encoding.UTF8.GetByteCount(_tokens[i])));
            }
        }

        _specials = specials.OrderByDescending(s => s.Item3).Select(s => (s.Item1, s.Item2)).ToArray();

        // end-of-generation ids
        var eog = new List<int>();
        foreach (int id in data.EosIds)
        {
            if (ValidId(id) is int v && !eog.Contains(v))
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
            if (_tokenToId.TryGetValue(name, out int id) && _types[id] is TokenType.Control or TokenType.UserDefined && !eog.Contains(id))
            {
                eog.Add(id);
            }
        }

        eog.Sort();
        _eogIds = eog.ToArray();
        _eogSet = [.. _eogIds];

        _wordCache = new ConcurrentDictionary<byte[], int[]>(ByteArrayComparer.Instance);
        _wordCacheSpan = _wordCache.GetAlternateLookup<ReadOnlySpan<byte>>();
    }

    // Symbol ids: vocabulary ids, then "virtual" ids for strings that occur in merges but not in the vocabulary
    // (llama.cpp works on strings, so such intermediate symbols still take part in merging).
    private MergeTable BuildMerges(string[] merges)
    {
        int vocab = _tokens.Length;
        var virtualIds = new Dictionary<string, int>(StringComparer.Ordinal);
        var virtualLookup = virtualIds.GetAlternateLookup<ReadOnlySpan<char>>();
        int nextVirtual = vocab;

        int Sym(ReadOnlySpan<char> s, bool create)
        {
            if (_tokenToIdSpan.TryGetValue(s, out int id))
            {
                return id;
            }

            if (virtualLookup.TryGetValue(s, out id))
            {
                return id;
            }

            if (!create)
            {
                return -1;
            }

            id = nextVirtual++;
            virtualIds[s.ToString()] = id;
            return id;
        }

        for (int b = 0; b < 256; b++)
        {
            _initialSymbol[b] = Sym([ByteEncoding.ByteToChar[b]], create: true);
        }

        // pass 1: make sure every component and result has a symbol id
        var seps = new int[merges.Length];
        char[] buf = new char[256];
        for (int i = 0; i < merges.Length; i++)
        {
            string m = merges[i];
            int sep = m.Length > 0 ? m.IndexOf(' ', 1) : -1;
            seps[i] = sep;
            if (sep < 0)
            {
                continue;
            }

            ReadOnlySpan<char> a = m.AsSpan(0, sep), b2 = m.AsSpan(sep + 1);
            Sym(a, true); Sym(b2, true);
            int total = a.Length + b2.Length;
            if (buf.Length < total)
            {
                buf = new char[total * 2];
            }

            a.CopyTo(buf); b2.CopyTo(buf.AsSpan(a.Length));
            Sym(buf.AsSpan(0, total), true);
        }

        // pass 2: fill the table in rank order
        var table = new MergeTable(merges.Length);
        for (int i = 0; i < merges.Length; i++)
        {
            int sep = seps[i];
            if (sep < 0)
            {
                continue;
            }

            string m = merges[i];
            ReadOnlySpan<char> a = m.AsSpan(0, sep), b2 = m.AsSpan(sep + 1);
            int total = a.Length + b2.Length;
            a.CopyTo(buf); b2.CopyTo(buf.AsSpan(a.Length));
            table.TryAdd(Sym(a, false), Sym(b2, false), i, Sym(buf.AsSpan(0, total), false));
        }
        return table;
    }

    // ---------------------------------------------------------------- vocabulary access

    public string? TokenString(int id) => (uint)id < (uint)_tokens.Length ? _tokens[id] : null;

    public TokenType GetTokenType(int id) => (uint)id < (uint)_types.Length ? _types[id] : TokenType.Undefined;

    public bool IsSpecial(int id) => GetTokenType(id) is TokenType.Control or TokenType.Unknown;

    public bool IsEndOfGeneration(int id) => _eogSet.Contains(id);

    public int? TokenId(string piece) => _tokenToId.TryGetValue(piece, out int id) ? id : null;

    public byte[] TokenBytes(int id)
    {
        if ((uint)id >= (uint)_tokens.Length)
        {
            return [];
        }

        return _tokenBytes[id] ??= ComputeTokenBytes(id);
    }

    private byte[] ComputeTokenBytes(int id)
    {
        string s = _tokens[id];
        switch (_types[id])
        {
            case TokenType.Control or TokenType.Unknown or TokenType.UserDefined:
                return Encoding.UTF8.GetBytes(s);
            case TokenType.Normal:
                var bytes = new List<byte>(s.Length);
                Span<byte> tmp = stackalloc byte[4];
                for (int i = 0; i < s.Length; i++)
                {
                    if (ByteEncoding.TryGetByte(s[i], out byte b))
                    {
                        bytes.Add(b);
                    }
                    else
                    {
                        int len = char.IsHighSurrogate(s[i]) && i + 1 < s.Length ? 2 : 1;
                        int n = Encoding.UTF8.GetBytes(s.AsSpan(i, len), tmp);
                        for (int k = 0; k < n; k++)
                        {
                            bytes.Add(tmp[k]);
                        }

                        i += len - 1;
                    }
                }
                return bytes.ToArray();
            default:
                return [];
        }
    }

    // ---------------------------------------------------------------- decode

    public byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false)
    {
        var ms = new MemoryStream(ids.Length * 4);
        foreach (int id in ids)
        {
            if (skipSpecial && IsSpecial(id))
            {
                continue;
            }

            ms.Write(TokenBytes(id));
        }
        return ms.ToArray();
    }

    public string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false)
        => Encoding.UTF8.GetString(DecodeBytes(ids, skipSpecial));

    // ---------------------------------------------------------------- encode

    private readonly record struct Fragment(bool IsToken, int Start, int Length, int Id);

    public int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        var output = new IntList(Math.Max(16, text.Length / 3 + 4));
        if (addSpecial && AddBos && BosId is int bos)
        {
            output.Add(bos);
        }

        if (text.Length > 0)
        {
            var fragments = Partition(text, parseSpecial);
            foreach (var f in fragments)
            {
                if (f.IsToken)
                {
                    output.Add(f.Id);
                }
                else
                {
                    EncodeRaw(text.AsSpan(f.Start, f.Length), ref output);
                }
            }
        }

        if (addSpecial && AddEos && EosId is int eos)
        {
            output.Add(eos);
        }

        return output.ToArray();
    }

    /// <summary>Port of llama.cpp's <c>tokenizer_st_partition</c>.</summary>
    private List<Fragment> Partition(string text, bool parseSpecial)
    {
        var buffer = new List<Fragment> { new(false, 0, text.Length, 0) };
        foreach (var (stext, sid) in _specials)
        {
            // user-defined tokens are matched even when parseSpecial is false
            if (!parseSpecial && _types[sid] is TokenType.Control or TokenType.Unknown)
            {
                continue;
            }

            if (text.AsSpan().IndexOf(stext, StringComparison.Ordinal) < 0)
            {
                continue;
            }

            var next = new List<Fragment>(buffer.Count + 4);
            foreach (var frag in buffer)
            {
                if (frag.IsToken) { next.Add(frag); continue; }
                int pos = frag.Start, end = frag.Start + frag.Length;
                while (true)
                {
                    int rel = pos < end ? text.AsSpan(pos, end - pos).IndexOf(stext, StringComparison.Ordinal) : -1;
                    if (rel < 0)
                    {
                        if (pos < end)
                        {
                            next.Add(new(false, pos, end - pos, 0));
                        }

                        break;
                    }
                    int match = pos + rel;
                    if (match > pos)
                    {
                        next.Add(new(false, pos, match - pos, 0));
                    }

                    next.Add(new(true, 0, 0, sid));
                    pos = match + stext.Length;
                }
            }
            buffer = next;
        }
        return buffer;
    }

    private void EncodeRaw(ReadOnlySpan<char> text, ref IntList output)
    {
        int n = text.Length;
        byte[] utf8 = ArrayPool<byte>.Shared.Rent(n * 3 + 4);
        int[] cpts = ArrayPool<int>.Shared.Rent(n + 1);
        int[] charStart = ArrayPool<int>.Shared.Rent(n + 2);
        try
        {
            // decode to code points + UTF-8 (lone surrogates become U+FFFD like Encoding.UTF8)
            int nc = 0, nb = 0;
            int[] byteStart = ArrayPool<int>.Shared.Rent(n + 2);
            try
            {
                for (int i = 0; i < n;)
                {
                    int cp;
                    int clen = 1;
                    char ch = text[i];
                    if (char.IsHighSurrogate(ch) && i + 1 < n && char.IsLowSurrogate(text[i + 1])) { cp = char.ConvertToUtf32(ch, text[i + 1]); clen = 2; }
                    else if (char.IsSurrogate(ch))
                    {
                        cp = 0xFFFD;
                    }
                    else
                    {
                        cp = ch;
                    }

                    cpts[nc] = cp; charStart[nc] = i; byteStart[nc] = nb; nc++;
                    if (cp < 0x80)
                    {
                        utf8[nb++] = (byte)cp;
                    }
                    else if (cp < 0x800) { utf8[nb++] = (byte)(0xC0 | (cp >> 6)); utf8[nb++] = (byte)(0x80 | (cp & 0x3F)); }
                    else if (cp < 0x10000) { utf8[nb++] = (byte)(0xE0 | (cp >> 12)); utf8[nb++] = (byte)(0x80 | ((cp >> 6) & 0x3F)); utf8[nb++] = (byte)(0x80 | (cp & 0x3F)); }
                    else { utf8[nb++] = (byte)(0xF0 | (cp >> 18)); utf8[nb++] = (byte)(0x80 | ((cp >> 12) & 0x3F)); utf8[nb++] = (byte)(0x80 | ((cp >> 6) & 0x3F)); utf8[nb++] = (byte)(0x80 | (cp & 0x3F)); }
                    i += clen;
                }
                charStart[nc] = n; byteStart[nc] = nb;

                var lengths = new List<int>(nc / 4 + 4);
                _splitter.Split(new SplitInput(text, cpts.AsSpan(0, nc), charStart.AsSpan(0, nc + 1)), lengths);

                int c = 0;
                foreach (int len in lengths)
                {
                    int b0 = byteStart[c], b1 = byteStart[c + len];
                    c += len;
                    EncodeWord(utf8.AsSpan(b0, b1 - b0), ref output);
                }
            }
            finally { ArrayPool<int>.Shared.Return(byteStart); }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(utf8);
            ArrayPool<int>.Shared.Return(cpts);
            ArrayPool<int>.Shared.Return(charStart);
        }
    }

    private void EncodeWord(ReadOnlySpan<byte> word, ref IntList output)
    {
        if (word.Length == 1 && _initialSymbol[word[0]] < _tokens.Length) { output.Add(_initialSymbol[word[0]]); return; }

        bool cacheable = word.Length <= MaxCachedWordBytes;
        if (cacheable && _wordCacheSpan.TryGetValue(word, out int[]? cached)) { output.AddRange(cached); return; }

        var local = new IntList(8);
        BpeWord(word, ref local);
        var result = local.ToArray();
        output.AddRange(result);
        if (cacheable && _wordCache.Count < MaxCacheEntries)
        {
            _wordCache.TryAdd(word.ToArray(), result);
        }
    }

    private void BpeWord(ReadOnlySpan<byte> word, ref IntList output)
    {
        int n = word.Length;
        if (_ignoreMerges)
        {
            Span<char> chars = n <= 256 ? stackalloc char[n] : new char[n];
            for (int i = 0; i < n; i++)
            {
                chars[i] = ByteEncoding.ByteToChar[word[i]];
            }

            if (_tokenToIdSpan.TryGetValue(chars, out int whole)) { output.Add(whole); return; }
        }

        int vocab = _tokens.Length;
        int[]? rented = null;
        Span<int> mem = n <= StackWordLimit ? stackalloc int[n * 5] : (rented = ArrayPool<int>.Shared.Rent(n * 5)).AsSpan(0, n * 5);
        try
        {
            Span<int> ids = mem[..n], len = mem.Slice(n, n), next = mem.Slice(2 * n, n), rankAt = mem.Slice(3 * n, n), resAt = mem.Slice(4 * n, n);
            for (int i = 0; i < n; i++)
            {
                ids[i] = _initialSymbol[word[i]];
                len[i] = 1;
                next[i] = i + 1 < n ? i + 1 : -1;
            }

            if (n > 1)
            {
                if (n <= HeapThreshold)
                {
                    MergeScan(ids, len, next, rankAt, resAt);
                }
                else
                {
                    MergeHeap(ids, len, next);
                }
            }

            int pos = 0;
            for (int i = 0; i != -1; i = next[i])
            {
                int id = ids[i];
                if (id < vocab)
                {
                    output.Add(id);
                }
                else
                {
                    // symbol text is not a vocabulary entry: fall back to its individual bytes
                    for (int k = 0; k < len[i]; k++)
                    {
                        int bid = _initialSymbol[word[pos + k]];
                        if (bid < vocab)
                        {
                            output.Add(bid);
                        }
                    }
                }
                pos += len[i];
            }
        }
        finally { if (rented != null)
            {
                ArrayPool<int>.Shared.Return(rented);
            }
        }
    }

    private void Refresh(int i, Span<int> ids, Span<int> next, Span<int> rankAt, Span<int> resAt)
    {
        int j = next[i];
        if (j >= 0 && _merges.TryGet(ids[i], ids[j], out int r, out int res)) { rankAt[i] = r; resAt[i] = res; }
        else
        {
            rankAt[i] = int.MaxValue;
        }
    }

    private void MergeScan(Span<int> ids, Span<int> len, Span<int> next, Span<int> rankAt, Span<int> resAt)
    {
        int n = ids.Length;
        for (int i = 0; i < n; i++)
        {
            Refresh(i, ids, next, rankAt, resAt);
        }

        int[]? prev = null;
        Span<int> prevSpan = n <= StackWordLimit ? stackalloc int[n] : (prev = new int[n]);
        for (int i = 0; i < n; i++)
        {
            prevSpan[i] = i - 1;
        }

        while (true)
        {
            int best = -1, bestRank = int.MaxValue;
            for (int i = 0; i != -1; i = next[i])
            {
                if (rankAt[i] < bestRank) { bestRank = rankAt[i]; best = i; }
            }

            if (best < 0)
            {
                break;
            }

            int right = next[best];
            ids[best] = resAt[best];
            len[best] += len[right];
            next[best] = next[right];
            if (next[right] >= 0)
            {
                prevSpan[next[right]] = best;
            }

            Refresh(best, ids, next, rankAt, resAt);
            if (prevSpan[best] >= 0)
            {
                Refresh(prevSpan[best], ids, next, rankAt, resAt);
            }
        }
    }

    private readonly record struct Bigram(int Left, int Right, int IdL, int IdR, int Result);

    private void Push(PriorityQueue<Bigram, long> queue, Span<int> ids, int l, int r)
    {
        if (l < 0 || r < 0)
        {
            return;
        }

        if (_merges.TryGet(ids[l], ids[r], out int rank, out int res))
        {
            queue.Enqueue(new Bigram(l, r, ids[l], ids[r], res), ((long)rank << 32) | (uint)l);
        }
    }

    private void MergeHeap(Span<int> ids, Span<int> len, Span<int> next)
    {
        int n = ids.Length;
        var prev = new int[n];
        var alive = new bool[n];
        for (int i = 0; i < n; i++) { prev[i] = i - 1; alive[i] = true; }
        var queue = new PriorityQueue<Bigram, long>(n);

        for (int i = 0; i + 1 < n; i++)
        {
            Push(queue, ids, i, i + 1);
        }

        while (queue.TryDequeue(out var bg, out _))
        {
            if (!alive[bg.Left] || !alive[bg.Right] || ids[bg.Left] != bg.IdL || ids[bg.Right] != bg.IdR || next[bg.Left] != bg.Right)
            {
                continue;
            }

            ids[bg.Left] = bg.Result;
            len[bg.Left] += len[bg.Right];
            alive[bg.Right] = false;
            next[bg.Left] = next[bg.Right];
            if (next[bg.Right] >= 0)
            {
                prev[next[bg.Right]] = bg.Left;
            }

            Push(queue, ids, prev[bg.Left], bg.Left);
            Push(queue, ids, bg.Left, next[bg.Left]);
        }
    }

    // ---------------------------------------------------------------- helpers

    private struct IntList
    {
        private int[] _buf;
        private int _count;
        public IntList(int capacity) { _buf = new int[capacity]; _count = 0; }
        public void Add(int v)
        {
            if (_count == _buf.Length)
            {
                Array.Resize(ref _buf, _buf.Length * 2);
            }

            _buf[_count++] = v;
        }
        public void AddRange(ReadOnlySpan<int> vs)
        {
            if (_count + vs.Length > _buf.Length)
            {
                Array.Resize(ref _buf, Math.Max(_buf.Length * 2, _count + vs.Length));
            }

            vs.CopyTo(_buf.AsSpan(_count));
            _count += vs.Length;
        }
        public int[] ToArray() => _buf.AsSpan(0, _count).ToArray();
    }

    private sealed class ByteArrayComparer : IEqualityComparer<byte[]>, IAlternateEqualityComparer<ReadOnlySpan<byte>, byte[]>
    {
        public static readonly ByteArrayComparer Instance = new();
        public bool Equals(byte[]? x, byte[]? y) => x is not null && y is not null && x.AsSpan().SequenceEqual(y);
        public int GetHashCode(byte[] obj) => GetHashCode((ReadOnlySpan<byte>)obj);
        public bool Equals(ReadOnlySpan<byte> alternate, byte[] other) => alternate.SequenceEqual(other);
        public int GetHashCode(ReadOnlySpan<byte> alternate)
        {
            var h = new HashCode();
            h.AddBytes(alternate);
            return h.ToHashCode();
        }
        public byte[] Create(ReadOnlySpan<byte> alternate) => alternate.ToArray();
    }
}
