namespace Sharplight.Tokenizers;

/// <summary>
/// A text tokenizer over a fixed vocabulary. Implementations are immutable after construction and safe to use from several threads;
/// arrays they return are fresh unless stated otherwise.
/// </summary>
public interface ITokenizer
{
    /// <summary>Number of token ids (ids are 0..VocabSize-1).</summary>
    public int VocabSize { get; }

    /// <summary>Id of the beginning-of-sequence token, or null if the vocabulary has none.</summary>
    public int? BosId { get; }

    /// <summary>Id of the end-of-sequence token, or null if the vocabulary has none.</summary>
    public int? EosId { get; }

    /// <summary>Id of the padding token, or null if the vocabulary has none.</summary>
    public int? PadId { get; }

    /// <summary>Whether <see cref="Encode"/> with addSpecial=true prepends BOS.</summary>
    public bool AddBos { get; }

    /// <summary>Whether <see cref="Encode"/> with addSpecial=true appends EOS.</summary>
    public bool AddEos { get; }

    /// <summary>
    /// Tokenizes text. When <paramref name="addSpecial"/> is set, BOS/EOS are added as configured by the model.
    /// When <paramref name="parseSpecial"/> is set, control tokens (e.g. &lt;|im_start|&gt;) in the text are
    /// matched literally and emitted as their single id (user-defined tokens are always matched).
    /// </summary>
    /// <param name="text">The text to tokenize.</param>
    /// <param name="addSpecial">Add the BOS/EOS tokens the model is configured to add.</param>
    /// <param name="parseSpecial">Match control tokens occurring literally in the text.</param>
    /// <returns>The token ids; empty text yields only the added special tokens.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is null.</exception>
    public int[] Encode(string text, bool addSpecial = true, bool parseSpecial = true);

    /// <summary>Decodes ids to text. Invalid UTF-8 produced by the byte sequence is replaced with U+FFFD. With skipSpecial, control tokens render as nothing.</summary>
    /// <param name="ids">Token ids; ids outside the vocabulary render as nothing.</param>
    /// <param name="skipSpecial">Hide control and unknown tokens (see <see cref="IsSpecial"/>).</param>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecial = false);

    /// <summary>Decodes ids to raw bytes (no UTF-8 repair).</summary>
    /// <param name="ids">Token ids; ids outside the vocabulary render as nothing.</param>
    /// <param name="skipSpecial">Hide control and unknown tokens (see <see cref="IsSpecial"/>).</param>
    public byte[] DecodeBytes(ReadOnlySpan<int> ids, bool skipSpecial = false);

    /// <summary>
    /// The bytes a token renders to (control tokens render as their literal text; unused tokens as empty).
    /// The returned array may be shared and must not be modified.
    /// </summary>
    public byte[] TokenBytes(int id);

    /// <summary>The raw vocabulary string of a token (for byte-level BPE: the GPT-2 byte-mapped form), or null if out of range.</summary>
    public string? TokenString(int id);

    /// <summary>The classification of a token; <see cref="TokenType.Undefined"/> for ids outside the vocabulary.</summary>
    public TokenType GetTokenType(int id);

    /// <summary>True for control (and unknown) tokens, which <c>skipSpecial</c> hides.</summary>
    public bool IsSpecial(int id);

    /// <summary>True when generation should stop after emitting this token (EOS, end-of-turn, ...); see <see cref="EndOfGenerationIds"/>.</summary>
    public bool IsEndOfGeneration(int id);

    /// <summary>All end-of-generation token ids (declared by the model plus well-known names found in the vocabulary), sorted ascending.</summary>
    public IReadOnlyList<int> EndOfGenerationIds { get; }

    /// <summary>Looks up a token by its vocabulary string (e.g. "&lt;|im_end|&gt;"). Null if absent.</summary>
    public int? TokenId(string piece);
}
