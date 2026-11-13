namespace Sharplight.Tokenizers;

public interface ITokenizer
{
    /// <summary>Number of token ids (ids are 0..VocabSize-1).</summary>
    int VocabSize { get; }

    int? BosId { get; }
    int? EosId { get; }
    int? PadId { get; }

    /// <summary>Whether <see cref="Encode"/> with addSpecial=true prepends BOS.</summary>
    bool AddBos { get; }

    /// <summary>Whether <see cref="Encode"/> with addSpecial=true appends EOS.</summary>
    bool AddEos { get; }

    /// <summary>
    /// Tokenizes text. When <paramref name="addSpecial"/> is set, BOS/EOS are added as configured by the model.
    /// When <paramref name="parseSpecial"/> is set, control tokens (e.g. &lt;|im_start|&gt;) in the text are
    /// matched literally and emitted as their single id (user-defined tokens are always matched).
    /// </summary>
    int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true);

    /// <summary>Decodes ids to text. Invalid UTF-8 produced by the byte sequence is replaced with U+FFFD. With skipSpecial, control tokens render as nothing.</summary>
    string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false);

    /// <summary>Decodes ids to raw bytes (no UTF-8 repair).</summary>
    byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false);

    /// <summary>
    /// The bytes a token renders to (control tokens render as their literal text; unused tokens as empty).
    /// The returned array is shared and must not be modified.
    /// </summary>
    byte[] TokenBytes(int id);

    /// <summary>The raw vocabulary string of a token (for byte-level BPE: the GPT-2 byte-mapped form), or null if out of range.</summary>
    string? TokenString(int id);

    TokenType GetTokenType(int id);

    /// <summary>True for control (and unknown) tokens, which <c>skipSpecial</c> hides.</summary>
    bool IsSpecial(int id);

    bool IsEndOfGeneration(int id);
    IReadOnlyList<int> EndOfGenerationIds { get; }

    /// <summary>Looks up a token by its vocabulary string (e.g. "&lt;|im_end|&gt;"). Null if absent.</summary>
    int? TokenId(string piece);
}
