using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>Assembles the <see cref="ModelDefinition"/> of the layer-stack adapters (hybrid, short-conv, RWKV) from their graphs and states.</summary>
internal static class RecurrentModelDefinition
{
    /// <summary>Creates the definition; every such model uses one position axis, the source's end-of-sequence ids and the shared metadata.</summary>
    public static ModelDefinition Create(ModelSource src, int layerCount, int hiddenSize, IReadOnlyDictionary<string, ComputeGraph> graphs, IReadOnlyList<StateDefinition> states)
    {
        var eos = new List<int>();
        if (src.Has(CanonicalKeys.TokenizerEosId))
        {
            eos.Add(src.Get<int>(CanonicalKeys.TokenizerEosId));
        }

        if (src.Has(CanonicalKeys.TokenizerEotId))
        {
            eos.Add(src.Get<int>(CanonicalKeys.TokenizerEotId));
        }

        var vocabulary = src.Weight("token_embedding.weight")!.Shape[0];
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = graphs,
            States = states,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Properties = new Dictionary<string, object>
            {
                [WellKnown.VocabSize] = vocabulary.IsSymbolic ? 0 : (int)vocabulary.Fixed,
                [WellKnown.ContextLength] = src.Arch("context_length", 32768),
                [WellKnown.EmbeddingLength] = hiddenSize,
                [WellKnown.PositionAxes] = 1,
                [WellKnown.EosTokenIds] = eos.ToArray(),
                ["n_layer"] = layerCount,
            },
        };
    }
}
