using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>A built shared-KV drafter (<see cref="SharedKvDraftHead.Build"/>).</summary>
/// <param name="Graph">The <see cref="WellKnown.DraftGraph"/>.</param>
/// <param name="States">The drafter's KV caches.</param>
/// <param name="Layers">The drafter's layer count.</param>
/// <param name="Weights">The drafter's weights, named under <see cref="SharedKvDraftHead.WeightPrefix"/>.</param>
internal sealed record SharedKvDraft(ComputeGraph Graph, IReadOnlyList<StateDefinition> States, int Layers, IReadOnlyList<WeightInfo> Weights);
