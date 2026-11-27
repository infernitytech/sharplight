using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>Assembles the <see cref="ModelDefinition"/> of the layer-stack adapters (hybrid, short-conv, RWKV) from their graphs and states.</summary>
internal static class RecurrentModelDefinition
{
    /// <summary>
    /// Creates the definition; every such model uses one position axis, the source's end-of-sequence ids and the shared metadata. A vision tower
    /// (<paramref name="vision"/>) adds its weights (from <paramref name="visionSource"/> when that is an auxiliary source) and properties.
    /// </summary>
    public static ModelDefinition Create(ModelSource src, int layerCount, int hiddenSize, IReadOnlyDictionary<string, ComputeGraph> graphs, IReadOnlyList<StateDefinition> states,
        IReadOnlyDictionary<string, WeightInfo>? weights = null, Vision.VisionParameters? vision = null, ModelSource? visionSource = null)
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
        var props = new Dictionary<string, object>
        {
            [WellKnown.VocabSize] = vocabulary.IsSymbolic ? 0 : (int)vocabulary.Fixed,
            [WellKnown.ContextLength] = src.Arch("context_length", 32768),
            [WellKnown.EmbeddingLength] = hiddenSize,
            [WellKnown.PositionAxes] = 1,
            [WellKnown.EosTokenIds] = eos.ToArray(),
            ["n_layer"] = layerCount,
        };
        vision?.WriteProperties(props);
        var owned = new List<IDisposable>(src.OwnedResources);
        if (visionSource is not null && !ReferenceEquals(visionSource, src))
        {
            owned.AddRange(visionSource.OwnedResources);
        }

        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = weights ?? src.Weights,
            Graphs = graphs,
            States = states,
            SourceMetadata = src.Metadata,
            OwnedResources = owned,
            Properties = props,
        };
    }
}
