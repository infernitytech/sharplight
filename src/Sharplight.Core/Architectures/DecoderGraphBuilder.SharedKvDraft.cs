using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

internal sealed partial class DecoderGraphBuilder
{
    /// <summary>The KV-source layers of a shared-KV drafter (one per attention kind its layers use), in ascending order.</summary>
    public int[] KvSourceLayers => [.. Enumerable.Range(0, _hp.NLayer).Select(_hp.KvSourceOf).Distinct().Order()];

    /// <summary>
    /// The <see cref="WellKnown.DraftGraph"/> of a drafter whose layers attend its target's keys and values (see <see cref="SharedKvDraftHead"/>),
    /// built on this builder's (drafter) source. The hidden-state input is [T, H_target + Σ kv]: the target's final hidden state (or the head's
    /// own projected output) followed by the keys and values of <see cref="KvSourceLayers"/> as the target computed them, each row as
    /// [K (kv heads · head size) | V (kv heads · value size)]. They are written to this builder's KV caches at the row's position, and the
    /// queries run at the next position (the token the row carries).
    /// </summary>
    /// <param name="targetEmbedding">The target's token embedding (the drafter reads the target's embedding of each row's token).</param>
    /// <param name="targetEmbeddingScale">The target's embedding scale.</param>
    /// <param name="targetHidden">The target's hidden size.</param>
    /// <param name="centroidTopK">Centroids kept by a centroid-routed LM head (<see cref="CentroidLogitMask"/>), when the drafter has one.</param>
    public ComputeGraph BuildSharedKvDraft(WeightInfo targetEmbedding, float targetEmbeddingScale, int targetHidden, int centroidTopK)
    {
        const string targetEmbeddingKey = "target.token_embedding.weight";
        var b = new GraphBuilder(WellKnown.DraftGraph, name => name == targetEmbeddingKey ? targetEmbedding : _src.Weight(name));
        var T = Dim.Sym(WellKnown.TokenSymbol);
        var sources = KvSourceLayers;
        long kvWidth = sources.Sum(il => (long)_hp.KvHeadsOf(il) * (_hp.HeadDimOf(il) + _hp.ValueDimOf(il)));
        Value x;
        Value positions;
        using (b.Group("input"))
        {
            var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
            var hiddenIn = b.Input("hidden_states", InputRole.HiddenStates, DataType.F32, new Shape(T, targetHidden + kvWidth));
            positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, _hp.PositionAxes));
            var embedded = b.Gather(b.Weight(targetEmbeddingKey), tokens, "draft_embd");
            if (targetEmbeddingScale != 1f)
            {
                embedded = b.Scale(embedded, targetEmbeddingScale);
            }

            x = b.Linear(b.Concat(1, embedded, b.Slice(hiddenIn, 1, 0, targetHidden)), "input_projection.weight", name: "draft_input_projection");
            _suppliedKv = [];
            long offset = targetHidden;
            foreach (int il in sources)
            {
                long keyWidth = (long)_hp.KvHeadsOf(il) * _hp.HeadDimOf(il);
                long valueWidth = (long)_hp.KvHeadsOf(il) * _hp.ValueDimOf(il);
                var k = b.Reshape(b.Slice(hiddenIn, 1, offset, keyWidth), 0, _hp.KvHeadsOf(il), _hp.HeadDimOf(il));
                var v = b.Reshape(b.Slice(hiddenIn, 1, offset + keyWidth, valueWidth), 0, _hp.KvHeadsOf(il), _hp.ValueDimOf(il));
                _suppliedKv[il] = (k, v);
                offset += keyWidth + valueWidth;
            }

            // The row at position p holds the target's keys and values of p and carries the token at p + 1, whose queries rotate by p + 1.
            positions = b.Add(positions, b.Constant("draft_query_offset", DataType.I32, new Shape(1), BitConverter.GetBytes(1)), "draft_query_positions");
        }

        var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
        x = Layers(b, x, positions, FrequencyFactors(b), perLayer: null, rowSpans: null);
        _suppliedKv = null;

        using (b.Group("output"))
        {
            var hidden = _norm.Apply(b, x, "output_norm.weight", _hp.Eps, "result_norm");
            var rows = b.Gather(hidden, outRows, "selected_rows");
            var logits = Logits(b, rows);
            if (b.TryWeight("centroids.weight") is { } centroids)
            {
                logits = CentroidLogitMask.Apply(b, logits, rows, centroids, b.Weight("centroid_of_token"), centroidTopK);
            }

            b.Output(WellKnown.LogitsOutput, logits);
            b.Output(WellKnown.HiddenOutput, b.Linear(hidden, "output_projection.weight", name: "draft_output_projection"));
        }

        return b.Build();
    }
}
