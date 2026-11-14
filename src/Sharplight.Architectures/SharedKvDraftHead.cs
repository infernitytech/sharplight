using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// A drafter whose layers compute only queries and attend its target's keys and values (canonical architecture <c>shared_kv_draft</c>, a separate
/// checkpoint loaded as the <see cref="Vision.AuxiliaryRoles.Draft"/> source of a <c>decoder</c> target; Gemma 4 assistants) as the target's
/// <see cref="WellKnown.DraftGraph"/>. Every draft of a step is made at the position of the token after the target's last verified row:
/// x = input_projection(concat(target_embed(token), h)), where h is the target's final hidden state at the row before (first draft) or the head's
/// own output_projection(output_norm(x_L)) (further drafts); its decoder layers (the <c>decoder</c> vocabulary, every layer sharing the KV cache of
/// the first layer of its attention kind) attend the keys and values of the target's last layer of the same kind, and the head's (tied) LM head,
/// optionally routed through centroids (<see cref="CentroidLogitMask"/>), predicts the next token.
/// </summary>
/// <remarks>
/// <para>
/// The target exports [final hidden state | K, V of the matching target layers] (<see cref="FeaturesOutput"/>); the head keeps its own copy of those
/// keys and values (one cache per attention kind, written from each observed row) and is <see cref="WellKnown.DraftAnchored"/>: further drafts
/// replace its last row. The queries of a row at position p rotate by p + 1. A sliding layer sees the last <c>sliding_window</c> keys up to p
/// (the importers turn the reference implementation's inclusive radius into this count).
/// </para>
/// <para>
/// Hyper-parameters: the <c>decoder</c> keys under <c>shared_kv_draft.</c> plus <c>shared_kv_draft.target_hidden_size</c> and, with a centroid head,
/// <c>shared_kv_draft.centroid_top_k</c>. Weights: the decoder layers without key / value projections, <c>output_norm.weight</c>,
/// <c>token_embedding.weight</c> (the LM head), <c>input_projection.weight</c> [H, 2·H_target], <c>output_projection.weight</c> [H_target, H],
/// optional <c>rope.freq_factors</c>, optional <c>centroids.weight</c> [C, H] with <c>centroid_of_token</c> (i32 [V]).
/// </para>
/// </remarks>
public static class SharedKvDraftHead
{
    /// <summary>Canonical architecture id of shared-KV drafter sources.</summary>
    public const string Architecture = "shared_kv_draft";

    /// <summary>The decode-graph output the drafter reads: [T, H_target + Σ kv], the final hidden state and the exported keys and values.</summary>
    public const string FeaturesOutput = "hidden_kv";

    /// <summary>Prefix of the drafter's weights and states in the target's definition.</summary>
    public const string WeightPrefix = "draft.";

    /// <summary>Whether <paramref name="source"/> is a shared-KV drafter.</summary>
    public static bool IsDrafter(ModelSource? source) => source is not null && source.Architecture == Architecture;

    /// <summary>
    /// The target layers whose keys and values the drafter reads, in the order of its KV-source layers: for each, the target's last layer of
    /// the same attention kind (sliding or full), resolved to the layer that computes its keys and values.
    /// </summary>
    /// <exception cref="InvalidDataException">The target has no layer of a kind, or its head shapes differ from the drafter's.</exception>
    internal static int[] TargetKvLayers(ModelSource drafter, DecoderHyperParameters target)
    {
        var hp = DecoderHyperParameters.Read(drafter);
        int width = drafter.Arch("target_hidden_size", target.NEmbd);
        if (width != target.NEmbd)
        {
            throw new InvalidDataException($"{Architecture}: the drafter was trained for a target of width {width}, the target's is {target.NEmbd}.");
        }

        var sources = Enumerable.Range(0, hp.NLayer).Select(hp.KvSourceOf).Distinct().Order();
        return [.. sources.Select(il =>
        {
            int last = Enumerable.Range(0, target.NLayer).LastOrDefault(t => target.Sliding(t) == hp.Sliding(il) && target.KvHeadsOf(t) > 0, -1);
            int layer = last < 0 ? -1 : target.KvSourceOf(last);
            if (layer < 0 || target.HeadDimOf(layer) != hp.HeadDimOf(il) || target.KvHeadsOf(layer) != hp.KvHeadsOf(il) || target.ValueDimOf(layer) != hp.ValueDimOf(il))
            {
                throw new InvalidDataException($"{Architecture}: the target has no {(hp.Sliding(il) ? "sliding" : "full")}-attention layer with the drafter's key / value shape.");
            }

            return layer;
        })];
    }

    /// <summary>
    /// Builds the drafter's <see cref="WellKnown.DraftGraph"/>, its states (one KV cache per attention kind) and its weights under
    /// <see cref="WeightPrefix"/> (to merge into the target's weights).
    /// </summary>
    /// <param name="drafter">The drafter source.</param>
    /// <param name="options">Adapter options of the target (KV cache type).</param>
    /// <param name="targetEmbedding">The target's token embedding.</param>
    /// <param name="target">The target's hyper-parameters.</param>
    internal static SharedKvDraft Build(ModelSource drafter, AdapterOptions options, WeightInfo targetEmbedding, DecoderHyperParameters target)
    {
        var weights = drafter.Weights.ToDictionary(kv => kv.Key, kv => kv.Value.As(WeightPrefix + kv.Key), StringComparer.Ordinal);
        var prefixed = new ModelSource
        {
            Format = drafter.Format,
            Architecture = drafter.Architecture,
            Metadata = drafter.Metadata,
            Weights = weights,
            Path = drafter.Path,
        };
        var hp = DecoderHyperParameters.Read(prefixed);
        var builder = new DecoderGraphBuilder(prefixed, options, hp, statePrefix: WeightPrefix);
        var graph = builder.BuildSharedKvDraft(targetEmbedding, target.EmbeddingScale, target.NEmbd, drafter.Arch("centroid_top_k", 32));
        return new SharedKvDraft(graph, builder.States, hp.NLayer, weights.Values.ToList());
    }
}
