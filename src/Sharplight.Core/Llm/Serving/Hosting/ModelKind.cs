namespace Sharplight.Llm.Serving.Hosting;

/// <summary>What a registered model does. Chat models serve the completion APIs; the others are reserved for model types that other components register.</summary>
public enum ModelKind
{
    /// <summary>A generative chat / completion model.</summary>
    Chat,

    /// <summary>A text embedding model.</summary>
    Embedding,

    /// <summary>A reranking (cross-encoder) model.</summary>
    Reranker,

    /// <summary>A model of the task layer (speech recognition, speech synthesis, ...) whose client comes from the <c>TaskClientProvider</c>.</summary>
    Task,
}
