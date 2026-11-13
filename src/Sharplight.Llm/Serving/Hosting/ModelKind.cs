namespace Sharplight.Llm.Serving.Hosting;

/// <summary>What a registered model does. Chat models serve the completion APIs; the others are reserved for model types that other components register.</summary>
public enum ModelKind { Chat, Embedding, Reranker }
