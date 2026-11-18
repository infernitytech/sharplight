namespace Sharplight.Studio.Services.FirstRun;

/// <summary>What a starter model is for.</summary>
internal enum StarterRole
{
    /// <summary>The chat model, sized to the machine.</summary>
    Chat,

    /// <summary>Speech to text.</summary>
    Speech,

    /// <summary>Text embeddings for search.</summary>
    Embedding,
}
