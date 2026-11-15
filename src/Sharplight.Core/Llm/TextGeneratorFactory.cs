using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Llm;

/// <summary>
/// The built-in text-generation implementation of the task layer: every <see cref="ModelKinds.TextGeneration"/> model with a decoder
/// graph (autoregressive) or a denoising graph (masked diffusion) gets a <see cref="TextGenerator"/> over an <see cref="LlmModel"/>
/// view of the task model.
/// </summary>
internal sealed class TextGeneratorFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.TextGeneration;

    public Type ClientType => typeof(ITextGenerator);

    public bool CanCreate(TaskModel model)
        => model.Definition.Graphs.ContainsKey(WellKnown.DecodeGraph) || model.Definition.Graphs.ContainsKey(WellKnown.DenoiseGraph);

    public ITaskClient Create(TaskModel model) => new TextGenerator(LlmModel.FromTaskModel(model), WellKnown.DecodeGraph, ownsModel: true);
}
