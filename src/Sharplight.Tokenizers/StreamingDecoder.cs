using System.Text;

namespace Sharplight.Tokenizers;

/// <summary>
/// Incremental detokenizer: feed tokens one by one, get back only complete UTF-8 text.
/// A multi-byte character split across tokens is buffered until its last byte arrives.
/// Not thread-safe; use one instance per generation stream.
/// </summary>
public sealed class StreamingDecoder
{
    private readonly ITokenizer _tokenizer;
    private readonly bool _skipSpecial;
    private readonly Decoder _decoder = new UTF8Encoding(false, false).GetDecoder();
    private char[] _chars = new char[64];

    /// <summary>Creates a decoder for one generation stream.</summary>
    /// <param name="tokenizer">Supplies the bytes of each token.</param>
    /// <param name="skipSpecial">Drop control and unknown tokens instead of rendering their text.</param>
    /// <exception cref="ArgumentNullException"><paramref name="tokenizer"/> is null.</exception>
    public StreamingDecoder(ITokenizer tokenizer, bool skipSpecial = false)
    {
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _skipSpecial = skipSpecial;
    }

    /// <summary>Adds a token and returns any newly completed text (possibly empty).</summary>
    public string Push(int id)
    {
        if (_skipSpecial && _tokenizer.IsSpecial(id))
        {
            return "";
        }

        byte[] bytes = _tokenizer.TokenBytes(id);
        return Feed(bytes, flush: false);
    }

    /// <summary>Adds several tokens at once.</summary>
    public string Push(ReadOnlySpan<int> ids)
    {
        var sb = new StringBuilder();
        foreach (int id in ids)
        {
            sb.Append(Push(id));
        }

        return sb.ToString();
    }

    /// <summary>
    /// Ends the stream. Bytes of an incomplete trailing character (if any) are emitted as U+FFFD, and the decoder is reset.
    /// </summary>
    public string Flush() => Feed([], flush: true);

    /// <summary>Discards any buffered partial character.</summary>
    public void Reset() => _decoder.Reset();

    private string Feed(ReadOnlySpan<byte> bytes, bool flush)
    {
        int max = _decoder.GetCharCount(bytes, flush);
        if (_chars.Length < max)
        {
            _chars = new char[Math.Max(max, _chars.Length * 2)];
        }

        int n = _decoder.GetChars(bytes, _chars, flush);
        return n == 0 ? "" : new string(_chars, 0, n);
    }
}
