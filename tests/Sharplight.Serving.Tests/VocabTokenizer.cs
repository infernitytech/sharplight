using System.Text;
using Sharplight.Tokenizers;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Byte tokens 0..255 plus multi-byte text tokens and special tokens, then EOS: enough structure to exercise the token trie
/// (tokens that span grammar boundaries, shared prefixes, special tokens).
/// </summary>
public sealed class VocabTokenizer : ITokenizer
{
    private readonly string[] _words;
    private readonly string[] _special;

    public VocabTokenizer(string[] words, string[]? special = null)
    {
        _words = words;
        _special = special ?? [];
        Eos = 256 + _words.Length + _special.Length;
    }

    public int Eos { get; }
    public int VocabSize => Eos + 1;
    public int? BosId => null;
    public int? EosId => Eos;
    public int? PadId => null;
    public bool AddBos => false;
    public bool AddEos => false;

    public int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true) => [.. Encoding.UTF8.GetBytes(text).Select(b => (int)b)];

    public string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false) => Encoding.UTF8.GetString(DecodeBytes(ids, skipSpecial));

    public byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false)
    {
        var bytes = new List<byte>();
        foreach (int id in ids)
        {
            if (!(skipSpecial && IsSpecial(id)))
            {
                bytes.AddRange(TokenBytes(id));
            }
        }

        return [.. bytes];
    }

    public byte[] TokenBytes(int id)
    {
        if (id < 256)
        {
            return [(byte)id];
        }

        if (id < 256 + _words.Length)
        {
            return Encoding.UTF8.GetBytes(_words[id - 256]);
        }

        if (id < Eos)
        {
            return Encoding.UTF8.GetBytes(_special[id - 256 - _words.Length]);
        }

        return id == Eos ? "</s>"u8.ToArray() : [];
    }

    public string? TokenString(int id) => Encoding.UTF8.GetString(TokenBytes(id));

    public TokenType GetTokenType(int id) => id < 256 ? TokenType.Byte : IsSpecial(id) ? TokenType.Control : TokenType.Normal;

    public bool IsSpecial(int id) => id >= 256 + _words.Length;

    public bool IsEndOfGeneration(int id) => id == Eos;

    public IReadOnlyList<int> EndOfGenerationIds => [Eos];

    public int? TokenId(string piece)
    {
        for (int id = 0; id < VocabSize; id++)
        {
            if (TokenString(id) == piece && (id >= 256 || piece.Length == 1))
            {
                return id;
            }
        }

        return null;
    }
}
