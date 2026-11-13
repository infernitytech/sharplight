using System.Text;
using System.Text.RegularExpressions;

namespace Sharplight.Tokenizers.Unigram;

/// <summary>
/// Unigram language-model tokenizer ("unigram"; SentencePiece unigram models such as XLM-RoBERTa, bge-m3, multilingual-e5),
/// following Hugging Face's <c>Unigram</c> model with a <c>Metaspace</c> pre-tokenizer: text is normalized (NFKC, runs of
/// spaces collapsed), spaces become ▁ and a ▁ is prepended, the text is split before every ▁ and each word is segmented by
/// Viterbi search for the most probable sequence of vocabulary pieces (sum of piece log-probabilities, <c>tokenizer.scores</c>).
/// Characters no piece covers become the unknown token (consecutive unknowns fused into one).
/// </summary>
/// <remarks>The model's precompiled normalization map (SentencePiece "nmt_nfkc") is approximated by Unicode NFKC plus removal of
/// control characters, which agrees for ordinary text.</remarks>
public sealed partial class UnigramTokenizer : ITokenizer
{
    private const char Space = '▁';
    private const float UnkPenalty = 10f;

    private readonly string[] _tokens;
    private readonly float[] _scores;
    private readonly TokenType[] _types;
    private readonly Dictionary<string, int> _toId = new(StringComparer.Ordinal);
    private readonly UnigramTrieNode _root = new();
    private readonly SpecialTokens _specials;
    private readonly int _unk;
    private readonly float _unkScore;
    private readonly bool _addSpacePrefix;
    private readonly int[] _eog;

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
    public IReadOnlyList<int> EndOfGenerationIds => _eog;

    /// <summary>Creates the tokenizer from a "unigram" description (pieces with scores).</summary>
    public UnigramTokenizer(TokenizerData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        _tokens = data.Tokens;
        int n = _tokens.Length;
        _scores = data.Scores is { } sc && sc.Length >= n ? sc : throw new InvalidDataException("Unigram tokenizer needs a score per token (tokenizer.scores).");
        _types = new TokenType[n];
        float min = float.MaxValue;
        for (int i = 0; i < n; i++)
        {
            _types[i] = data.TokenTypes is { } tt && i < tt.Length && Enum.IsDefined((TokenType)tt[i]) ? (TokenType)tt[i] : TokenType.Normal;
            _toId.TryAdd(_tokens[i], i);
            if (_types[i] is TokenType.Normal or TokenType.UserDefined)
            {
                Insert(_tokens[i], i);
                min = Math.Min(min, _scores[i]);
            }
        }
        int? Valid(int? id) => id is int v && v >= 0 && v < n ? v : null;
        _unk = Valid(data.UnkId) ?? (_toId.TryGetValue("<unk>", out int u) ? u : 0);
        _unkScore = (min == float.MaxValue ? 0 : min) - UnkPenalty;
        BosId = Valid(data.BosId);
        EosId = Valid(data.EosId);
        PadId = Valid(data.PadId);
        AddBos = data.AddBos ?? BosId is not null;
        AddEos = data.AddEos ?? EosId is not null;
        _addSpacePrefix = data.AddSpacePrefix ?? true;
        _eog = EosId is int e ? [e] : [];
        _specials = new SpecialTokens(_tokens, i => _types[i]);
    }

    private void Insert(string piece, int id)
    {
        var node = _root;
        foreach (char c in piece)
        {
            node.Next ??= [];
            if (!node.Next.TryGetValue(c, out var nx))
            {
                node.Next[c] = nx = new UnigramTrieNode();
            }

            node = nx;
        }
        if (node.Id < 0)
        {
            node.Id = id;
        }
    }

    /// <inheritdoc/>
    public string? TokenString(int id) => (uint)id < (uint)_tokens.Length ? _tokens[id] : null;
    /// <inheritdoc/>
    public TokenType GetTokenType(int id) => (uint)id < (uint)_types.Length ? _types[id] : TokenType.Undefined;
    /// <inheritdoc/>
    public bool IsSpecial(int id) => GetTokenType(id) is TokenType.Control or TokenType.Unknown;
    /// <inheritdoc/>
    public bool IsEndOfGeneration(int id) => Array.IndexOf(_eog, id) >= 0;
    /// <inheritdoc/>
    public int? TokenId(string piece) => _toId.TryGetValue(piece, out int id) ? id : null;

    /// <inheritdoc/>
    public byte[] TokenBytes(int id)
        => (uint)id >= (uint)_tokens.Length ? [] : Encoding.UTF8.GetBytes(_types[id] == TokenType.Normal ? _tokens[id].Replace(Space, ' ') : _tokens[id]);

    /// <inheritdoc/>
    public byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false) => Encoding.UTF8.GetBytes(Decode(ids, skipSpecial));

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false)
    {
        var sb = new StringBuilder();
        foreach (int id in ids)
        {
            if ((uint)id >= (uint)_tokens.Length || skipSpecial && IsSpecial(id))
            {
                continue;
            }

            sb.Append(_types[id] == TokenType.Normal ? _tokens[id].Replace(Space, ' ') : _tokens[id]);
        }
        string s = sb.ToString();
        return _addSpacePrefix && s.StartsWith(' ') ? s[1..] : s;
    }

    /// <inheritdoc/>
    public int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        var output = new List<int>(text.Length / 3 + 4);
        if (addSpecial && AddBos && BosId is int bos)
        {
            output.Add(bos);
        }

        foreach (var (frag, id) in _specials.Split(text, parseSpecial))
        {
            if (id >= 0) { output.Add(id); continue; }
            string s = Normalize(frag);
            if (s.Length == 0)
            {
                continue;
            }

            s = s.Replace(' ', Space);
            if (_addSpacePrefix && s[0] != Space)
            {
                s = Space + s;
            }

            int start = 0;
            for (int i = 1; i <= s.Length; i++)
            {
                if (i == s.Length || s[i] == Space)
                {
                    EncodeWord(s.AsSpan(start, i - start), output);
                    start = i;
                }
            }
        }
        if (addSpecial && AddEos && EosId is int eos)
        {
            output.Add(eos);
        }

        return [.. output];
    }

    [GeneratedRegex(" {2,}")]
    private static partial Regex MultiSpace();

    private static string Normalize(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var r in text.Normalize(NormalizationForm.FormKC).EnumerateRunes())
        {
            if (r.Value is '\t' or '\n' or '\r' || Rune.IsWhiteSpace(r)) { sb.Append(' '); continue; }
            if (Rune.IsControl(r))
            {
                continue;
            }

            sb.Append(r.ToString());
        }
        return MultiSpace().Replace(sb.ToString(), " ");
    }

    /// <summary>Viterbi segmentation of one word (best total score; uncovered characters become a fused unknown token).</summary>
    private void EncodeWord(ReadOnlySpan<char> w, List<int> output)
    {
        int n = w.Length;
        var best = new float[n + 1];
        var from = new int[n + 1];
        var piece = new int[n + 1];
        Array.Fill(best, float.NegativeInfinity);
        best[0] = 0;
        for (int i = 0; i < n; i++)
        {
            if (float.IsNegativeInfinity(best[i]))
            {
                continue;
            }

            if (char.IsLowSurrogate(w[i]))
            {
                continue;
            }

            int charLen = char.IsHighSurrogate(w[i]) && i + 1 < n ? 2 : 1;
            bool single = false;
            var node = _root;
            for (int j = i; j < n && node.Next is not null && node.Next.TryGetValue(w[j], out node); j++)
            {
                if (node.Id < 0)
                {
                    continue;
                }

                if (j + 1 == i + charLen)
                {
                    single = true;
                }

                float s = best[i] + _scores[node.Id];
                if (s > best[j + 1]) { best[j + 1] = s; from[j + 1] = i; piece[j + 1] = node.Id; }
            }
            if (!single)
            {
                float s = best[i] + _unkScore;
                if (s > best[i + charLen]) { best[i + charLen] = s; from[i + charLen] = i; piece[i + charLen] = -1; }
            }
        }
        var ids = new List<int>();
        for (int e = n; e > 0; e = from[e])
        {
            ids.Add(piece[e]);
        }

        ids.Reverse();
        bool prevUnk = false;
        foreach (int id in ids)
        {
            if (id < 0) { if (!prevUnk) { output.Add(_unk); } prevUnk = true; continue; }
            output.Add(id);
            prevUnk = false;
        }
    }
}
