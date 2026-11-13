using System.Text;
using Sharplight.Tokenizers;

namespace Sharplight.Serving.Tests;

/// <summary>Byte-level tokenizer: ids 0..255 are bytes, 256 = BOS, 257 = EOS (end of generation).</summary>
public sealed class FakeTokenizer : ITokenizer
{
    public const int Bos = 256;
    public const int Eos = 257;
    public int VocabSize => 260;
    public int? BosId => Bos;
    public int? EosId => Eos;
    public int? PadId => null;
    public bool AddBos => false;
    public bool AddEos => false;
    public int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true)
        => [.. (addSpecial && AddBos ? [Bos] : Array.Empty<int>()), .. Encoding.UTF8.GetBytes(text).Select(b => (int)b)];
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false) => Encoding.UTF8.GetString(DecodeBytes(ids, skipSpecial));
    public byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false)
    {
        var l = new List<byte>();
        foreach (var id in ids)
        {
            if (!(skipSpecial && IsSpecial(id)))
            {
                l.AddRange(TokenBytes(id));
            }
        }

        return [.. l];
    }
    public byte[] TokenBytes(int id) => id < 256 ? [(byte)id] : id == Bos ? "<s>"u8.ToArray() : id == Eos ? "</s>"u8.ToArray() : [];
    public string? TokenString(int id) => id < 128 ? ((char)id).ToString() : id == Bos ? "<s>" : id == Eos ? "</s>" : null;
    public TokenType GetTokenType(int id) => id < 256 ? TokenType.Byte : TokenType.Control;
    public bool IsSpecial(int id) => id >= 256;
    public bool IsEndOfGeneration(int id) => id == Eos;
    public IReadOnlyList<int> EndOfGenerationIds { get; } = [Eos];
    public int? TokenId(string piece) => piece == "<s>" ? Bos : piece == "</s>" ? Eos : null;
}
