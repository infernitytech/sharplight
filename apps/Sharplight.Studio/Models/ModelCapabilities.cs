namespace Sharplight.Studio.Models;

/// <summary>What a model can do, as far as Sharplight runs it.</summary>
[Flags]
internal enum ModelCapabilities
{
    /// <summary>Nothing known.</summary>
    None = 0,

    /// <summary>Generates text with a chat template.</summary>
    Chat = 1,

    /// <summary>Has a vision tower the engine runs: images can be attached.</summary>
    Vision = 2,

    /// <summary>The chat template emits a reasoning (thinking) block.</summary>
    Reasoning = 4,

    /// <summary>The chat template renders tool definitions.</summary>
    Tools = 8,

    /// <summary>Produces embeddings.</summary>
    Embedding = 16,

    /// <summary>Has a multi-token-prediction draft head for self-speculative decoding.</summary>
    DraftHead = 32,
}
