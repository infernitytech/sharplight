using System.Globalization;
using System.Text;

namespace Sharplight.Tokenizers.WordPiece;

/// <summary>
/// WordPiece tokenizer ("wordpiece") of BERT-family encoders (BERT, MiniLM, bge, e5, nomic-embed, ...), following Hugging Face's
/// <c>BertNormalizer</c> + <c>BertPreTokenizer</c> + <c>WordPiece</c> model: control characters are removed and whitespace
/// unified, CJK ideographs isolated, accents stripped and text lowercased (when configured); words split on whitespace and
/// punctuation; each word is matched greedily longest-first against the vocabulary, continuation pieces carrying the
/// <c>##</c> prefix; a word that cannot be matched becomes the unknown token. Vocabulary strings are in the Hugging Face form
/// (<c>##</c> marks continuations).
/// </summary>
public sealed class WordPieceTokenizer : ITokenizer
{
    const string Prefix = "##";
    const int MaxWordChars = 100;

    readonly string[] _tokens;
    readonly TokenType[] _types;
    readonly Dictionary<string, int> _toId = new(StringComparer.Ordinal);
    readonly SpecialTokens _specials;
    readonly bool _lowercase, _stripAccents;
    readonly int _unk;

    public int VocabSize => _tokens.Length;
    public int? BosId { get; }
    public int? EosId { get; }
    public int? PadId { get; }
    public bool AddBos { get; }
    public bool AddEos { get; }
    public IReadOnlyList<int> EndOfGenerationIds { get; }

    public WordPieceTokenizer(TokenizerData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        _tokens = data.Tokens;
        _types = new TokenType[_tokens.Length];
        for (int i = 0; i < _tokens.Length; i++)
        {
            _types[i] = data.TokenTypes is { } tt && i < tt.Length && Enum.IsDefined((TokenType)tt[i]) ? (TokenType)tt[i] : TokenType.Normal;
            _toId.TryAdd(_tokens[i], i);
        }
        int? Valid(int? id) => id is int v && v >= 0 && v < _tokens.Length ? v : null;
        int? Named(string s) => _toId.TryGetValue(s, out int id) ? id : null;
        _unk = Valid(data.UnkId) ?? Named("[UNK]") ?? 0;
        if (_types[_unk] == TokenType.Normal) _types[_unk] = TokenType.Unknown;
        BosId = Valid(data.BosId) ?? Named("[CLS]");
        EosId = Valid(data.EosId) ?? Named("[SEP]");
        PadId = Valid(data.PadId) ?? Named("[PAD]");
        foreach (var s in new[] { "[CLS]", "[SEP]", "[PAD]", "[MASK]" })
            if (Named(s) is int id && _types[id] == TokenType.Normal) _types[id] = TokenType.Control;
        AddBos = data.AddBos ?? true;
        AddEos = data.AddEos ?? true;
        _lowercase = data.Lowercase ?? true;
        _stripAccents = data.StripAccents ?? _lowercase;
        EndOfGenerationIds = EosId is int e ? [e] : [];
        _specials = new SpecialTokens(_tokens, i => _types[i]);
    }

    public string? TokenString(int id) => (uint)id < (uint)_tokens.Length ? _tokens[id] : null;
    public TokenType GetTokenType(int id) => (uint)id < (uint)_types.Length ? _types[id] : TokenType.Undefined;
    public bool IsSpecial(int id) => GetTokenType(id) is TokenType.Control or TokenType.Unknown;
    public bool IsEndOfGeneration(int id) => EosId == id;
    public int? TokenId(string piece) => _toId.TryGetValue(piece, out int id) ? id : null;

    public byte[] TokenBytes(int id)
    {
        if ((uint)id >= (uint)_tokens.Length) return [];
        string t = _tokens[id];
        return Encoding.UTF8.GetBytes(_types[id] == TokenType.Normal && t.StartsWith(Prefix, StringComparison.Ordinal) ? t[Prefix.Length..] : " " + t);
    }

    public byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false) => Encoding.UTF8.GetBytes(Decode(ids, skipSpecial));

    public string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false)
    {
        var sb = new StringBuilder();
        foreach (int id in ids)
        {
            if ((uint)id >= (uint)_tokens.Length || skipSpecial && IsSpecial(id)) continue;
            string t = _tokens[id];
            if (_types[id] == TokenType.Normal && t.StartsWith(Prefix, StringComparison.Ordinal)) sb.Append(t, Prefix.Length, t.Length - Prefix.Length);
            else { if (sb.Length > 0) sb.Append(' '); sb.Append(t); }
        }
        return sb.ToString();
    }

    public int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        var output = new List<int>(text.Length / 4 + 4);
        if (addSpecial && AddBos && BosId is int bos) output.Add(bos);
        foreach (var (frag, id) in _specials.Split(text, parseSpecial))
        {
            if (id >= 0) { output.Add(id); continue; }
            foreach (var word in PreTokenize(Normalize(frag))) EncodeWord(word, output);
        }
        if (addSpecial && AddEos && EosId is int eos) output.Add(eos);
        return [.. output];
    }

    // ---- normalization (BertNormalizer) --------------------------------------------------------
    string Normalize(string text)
    {
        var sb = new StringBuilder(text.Length + 8);
        foreach (var rune in text.EnumerateRunes())
        {
            int c = rune.Value;
            if (c == 0 || c == 0xFFFD || IsControl(rune)) continue;
            if (IsWhitespace(rune)) { sb.Append(' '); continue; }
            if (IsChinese(c)) { sb.Append(' ').Append(rune.ToString()).Append(' '); continue; }
            sb.Append(rune.ToString());
        }
        string s = sb.ToString();
        if (_stripAccents)
        {
            s = s.Normalize(NormalizationForm.FormD);
            var nb = new StringBuilder(s.Length);
            foreach (var r in s.EnumerateRunes())
                if (Rune.GetUnicodeCategory(r) != UnicodeCategory.NonSpacingMark) nb.Append(r.ToString());
            s = nb.ToString();
        }
        if (_lowercase) s = s.ToLowerInvariant();
        return s;
    }

    static bool IsWhitespace(Rune r) => r.Value is '\t' or '\n' or '\r' || Rune.IsWhiteSpace(r);

    static bool IsControl(Rune r)
    {
        if (r.Value is '\t' or '\n' or '\r') return false;
        return Rune.GetUnicodeCategory(r) is UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.Surrogate
            or UnicodeCategory.PrivateUse or UnicodeCategory.OtherNotAssigned;
    }

    static bool IsChinese(int c) =>
        c is >= 0x4E00 and <= 0x9FFF or >= 0x3400 and <= 0x4DBF or >= 0x20000 and <= 0x2A6DF or >= 0x2A700 and <= 0x2B73F
          or >= 0x2B740 and <= 0x2B81F or >= 0x2B820 and <= 0x2CEAF or >= 0xF900 and <= 0xFAFF or >= 0x2F800 and <= 0x2FA1F;

    static bool IsPunctuation(Rune r)
    {
        int c = r.Value;
        if (c is >= 33 and <= 47 or >= 58 and <= 64 or >= 91 and <= 96 or >= 123 and <= 126) return true;
        return Rune.GetUnicodeCategory(r) is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation or UnicodeCategory.OpenPunctuation
            or UnicodeCategory.ClosePunctuation or UnicodeCategory.InitialQuotePunctuation or UnicodeCategory.FinalQuotePunctuation or UnicodeCategory.OtherPunctuation;
    }

    // ---- pre-tokenization (BertPreTokenizer): whitespace splits, punctuation is isolated -------------
    static List<string> PreTokenize(string s)
    {
        var words = new List<string>();
        var cur = new StringBuilder();
        void Flush() { if (cur.Length > 0) { words.Add(cur.ToString()); cur.Clear(); } }
        foreach (var r in s.EnumerateRunes())
        {
            if (Rune.IsWhiteSpace(r)) { Flush(); continue; }
            if (IsPunctuation(r)) { Flush(); words.Add(r.ToString()); continue; }
            cur.Append(r.ToString());
        }
        Flush();
        return words;
    }

    // ---- WordPiece model: greedy longest match ---------------------------------------------------------
    void EncodeWord(string word, List<int> output)
    {
        var runes = word.EnumerateRunes().Select(r => r.ToString()).ToArray();
        if (runes.Length > MaxWordChars) { output.Add(_unk); return; }
        int mark = output.Count;
        int start = 0;
        var sb = new StringBuilder();
        while (start < runes.Length)
        {
            int end = runes.Length, found = -1;
            while (start < end)
            {
                sb.Clear();
                if (start > 0) sb.Append(Prefix);
                for (int i = start; i < end; i++) sb.Append(runes[i]);
                if (_toId.TryGetValue(sb.ToString(), out int id)) { found = id; break; }
                end--;
            }
            if (found < 0)
            {
                output.RemoveRange(mark, output.Count - mark);
                output.Add(_unk);
                return;
            }
            output.Add(found);
            start = end;
        }
    }
}
