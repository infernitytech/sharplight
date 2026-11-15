using System.Text;
using System.Text.RegularExpressions;

namespace Sharplight.Tokenizers.Bpe;

/// <summary>
/// The byte-level BPE of CLIP (model "clip"; also the text encoders of Stable Diffusion): text is normalized (NFC, whitespace runs collapsed, lower
/// case), split with CLIP's pattern, every piece is mapped to the GPT-2 byte alphabet with the end-of-word marker <c>&lt;/w&gt;</c> on its last
/// character and merged by rank. Tokens are stored as in <c>vocab.json</c> (including the marker); the begin and end of text tokens are added around
/// the text.
/// </summary>
internal sealed partial class ClipBpeTokenizer : ITokenizer
{
    private const string EndOfWord = "</w>";

    private readonly string[] _tokens;
    private readonly Dictionary<string, int> _toId = new(StringComparer.Ordinal);
    private readonly Dictionary<(string, string), int> _ranks = new();
    private readonly Dictionary<string, int[]> _cache = new(StringComparer.Ordinal);
    private readonly int[] _eog;
    private readonly bool _lowercase;

    /// <summary>Creates the tokenizer from a "clip" description: <see cref="TokenizerData.Tokens"/> in id order and <see cref="TokenizerData.Merges"/> as "left right".</summary>
    /// <param name="data">The description.</param>
    public ClipBpeTokenizer(TokenizerData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        _tokens = data.Tokens;
        for (int i = 0; i < _tokens.Length; i++)
        {
            _toId.TryAdd(_tokens[i], i);
        }

        string[] merges = data.Merges ?? throw new InvalidDataException("The clip tokenizer needs merges (tokenizer.merges).");
        for (int i = 0; i < merges.Length; i++)
        {
            int space = merges[i].IndexOf(' ', StringComparison.Ordinal);
            if (space > 0)
            {
                _ranks.TryAdd((merges[i][..space], merges[i][(space + 1)..]), i);
            }
        }

        BosId = TokenizerSetup.ValidId(data.BosId, _tokens.Length) ?? TokenId("<|startoftext|>");
        EosId = TokenizerSetup.ValidId(data.EosId, _tokens.Length) ?? TokenId("<|endoftext|>");
        PadId = TokenizerSetup.ValidId(data.PadId, _tokens.Length) ?? EosId;
        AddBos = data.AddBos ?? true;
        AddEos = data.AddEos ?? true;
        _lowercase = data.Lowercase ?? true;
        _eog = EosId is int e ? [e] : [];
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
    public bool AddBos { get; }

    /// <inheritdoc/>
    public bool AddEos { get; }

    /// <inheritdoc/>
    public IReadOnlyList<int> EndOfGenerationIds => _eog;

    /// <inheritdoc/>
    public int? TokenId(string piece) => _toId.TryGetValue(piece, out int id) ? id : null;

    /// <inheritdoc/>
    public string? TokenString(int id) => (uint)id < (uint)_tokens.Length ? _tokens[id] : null;

    /// <inheritdoc/>
    public TokenType GetTokenType(int id) => id == BosId || id == EosId ? TokenType.Control : TokenType.Normal;

    /// <inheritdoc/>
    public bool IsSpecial(int id) => id == BosId || id == EosId;

    /// <inheritdoc/>
    public bool IsEndOfGeneration(int id) => id == EosId;

    /// <inheritdoc/>
    public byte[] TokenBytes(int id) => (uint)id < (uint)_tokens.Length ? Bytes(_tokens[id]) : [];

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

            bytes.AddRange(Bytes(_tokens[id]));
        }

        return [.. bytes];
    }

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false) => Encoding.UTF8.GetString(DecodeBytes(ids, skipSpecial));

    /// <inheritdoc/>
    public int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        string normalized = Whitespace().Replace(text.Normalize(NormalizationForm.FormC), " ").Trim();
        if (_lowercase)
        {
            normalized = normalized.ToLowerInvariant();
        }

        var output = new List<int>();
        if (addSpecial && AddBos && BosId is int bos)
        {
            output.Add(bos);
        }

        foreach (Match m in Pattern().Matches(normalized))
        {
            if (parseSpecial && (m.Value == "<|startoftext|>" || m.Value == "<|endoftext|>") && TokenId(m.Value) is int special)
            {
                output.Add(special);
                continue;
            }

            output.AddRange(EncodePiece(m.Value));
        }

        if (addSpecial && AddEos && EosId is int eos)
        {
            output.Add(eos);
        }

        return [.. output];
    }

    private int[] EncodePiece(string piece)
    {
        if (_cache.TryGetValue(piece, out var cached))
        {
            return cached;
        }

        var symbols = new List<string>();
        foreach (byte b in Encoding.UTF8.GetBytes(piece))
        {
            symbols.Add(ByteEncoding.ByteToChar[b].ToString());
        }

        symbols[^1] += EndOfWord;
        while (symbols.Count > 1)
        {
            int best = int.MaxValue;
            int at = -1;
            for (int i = 0; i + 1 < symbols.Count; i++)
            {
                if (_ranks.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < best)
                {
                    best = rank;
                    at = i;
                }
            }

            if (at < 0)
            {
                break;
            }

            symbols[at] += symbols[at + 1];
            symbols.RemoveAt(at + 1);
        }

        var ids = symbols.Select(s => _toId.TryGetValue(s, out int id) ? id : EosId ?? 0).ToArray();
        _cache[piece] = ids;
        return ids;
    }

    private static byte[] Bytes(string token)
    {
        string text = token.EndsWith(EndOfWord, StringComparison.Ordinal) ? token[..^EndOfWord.Length] + " " : token;
        var bytes = new List<byte>();
        foreach (char c in text)
        {
            if (c == ' ')
            {
                bytes.Add((byte)' ');
            }
            else if (ByteEncoding.TryGetByte(c, out byte b))
            {
                bytes.Add(b);
            }
        }

        return [.. bytes];
    }

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespace();

    [GeneratedRegex(@"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")]
    private static partial Regex Pattern();
}
