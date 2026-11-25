using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Llm.Embeddings;

/// <summary>
/// Creates <see cref="TextRerankerClient"/>s for models with a <see cref="WellKnown.ClassifyGraph"/>: registered for
/// <see cref="ModelKinds.TextReranking"/> models and for language models (<see cref="ModelKinds.TextGeneration"/>), whose adapters
/// build the classify graph next to the decoder (cross-encoders and decoder rerankers describe themselves as language models).
/// </summary>
/// <param name="kind">The model kind served.</param>
internal sealed class TextRerankerFactory(string kind) : ITaskClientFactory
{
    public string Kind => kind;

    public Type ClientType => typeof(ITextReranker);

    public bool CanCreate(TaskModel model) => model.Definition.Graphs.ContainsKey(WellKnown.ClassifyGraph);

    public ITaskClient Create(TaskModel model) => new TextRerankerClient(model);
}
