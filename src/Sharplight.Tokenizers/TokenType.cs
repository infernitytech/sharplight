namespace Sharplight.Tokenizers;

/// <summary>Token classification; the numeric values are the canonical <c>tokenizer.token_type</c> values of GGUF and Hugging Face metadata.</summary>
public enum TokenType
{
    /// <summary>No classification available (also returned for ids outside the vocabulary).</summary>
    Undefined = 0,

    /// <summary>An ordinary vocabulary piece.</summary>
    Normal = 1,

    /// <summary>The unknown-token placeholder; treated as a special (hidden) token.</summary>
    Unknown = 2,

    /// <summary>A control token such as BOS/EOS or a chat-template marker; only matched in text when special parsing is requested.</summary>
    Control = 3,

    /// <summary>A user-defined token that is always matched literally in text.</summary>
    UserDefined = 4,

    /// <summary>A reserved slot that renders as nothing.</summary>
    Unused = 5,

    /// <summary>A byte-fallback token of the form <c>&lt;0xXX&gt;</c>.</summary>
    Byte = 6,
}
