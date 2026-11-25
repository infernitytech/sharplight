using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// Creates <see cref="TextEmbedderClient"/>s for models with an <see cref="WellKnown.EmbedGraph"/>: registered for
/// <see cref="ModelKinds.TextEmbedding"/> models and for language models (<see cref="ModelKinds.TextGeneration"/>), whose adapters
/// build the embed graph next to the decoder (BERT-family encoders and Qwen3-Embedding describe themselves as language models).
/// </summary>
/// <param name="kind">The model kind served.</param>
internal sealed class TextEmbedderFactory(string kind) : ITaskClientFactory
{
    public string Kind => kind;

    public Type ClientType => typeof(ITextEmbedder);

    public bool CanCreate(TaskModel model) => model.Definition.Graphs.ContainsKey(WellKnown.EmbedGraph);

    public ITaskClient Create(TaskModel model) => new TextEmbedderClient(model);
}
