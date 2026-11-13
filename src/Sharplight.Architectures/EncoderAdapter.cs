using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Bidirectional transformer encoder (canonical arch <c>encoder</c>): BERT and its descendants (MiniLM, bge, e5, gte, ...),
/// RoBERTa / XLM-RoBERTa (bge-m3, bge-reranker, multilingual-e5) and nomic-bert (rotary, SwiGLU). Builds the packed-sequence
/// graphs <see cref="WellKnown.EmbedGraph"/> (pooled sentence embeddings) and, when the model has a classifier head,
/// <see cref="WellKnown.ClassifyGraph"/> (cross-encoder scores).
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>encoder.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>ffn_hidden_size</c>, <c>attention.head_count</c>,
/// <c>norm_eps</c> (LayerNorm epsilon), <c>context_length</c> (longest sequence; for learned positions at most the table size),
/// <c>mlp.activation</c> ("gelu" (erf, default), "gelu_tanh", "relu", "silu"), <c>mlp.gated</c> (default false: act(up); true:
/// up · act(gate), SwiGLU), <c>rope.base</c> / <c>rope.dim</c> / <c>rope.style</c> (rotary positions when there is no position table;
/// "neox" default).
/// Embedding behaviour comes from the canonical <see cref="CanonicalKeys.EmbeddingPooling"/> (default "cls") and
/// <see cref="CanonicalKeys.EmbeddingNormalize"/> (default true) keys; <see cref="AdapterOptions.EmbeddingPooling"/> /
/// <see cref="AdapterOptions.EmbeddingNormalize"/> override them.
/// <para>Weights: <c>token_embedding.weight</c> [V, H]; optional <c>position_embedding.weight</c> [P, H] (row p is position p, counted
/// from 0 in every sequence — offsets such as RoBERTa's padding_idx + 1 are removed by the importer); optional
/// <c>token_type_embedding.weight</c> [types, H]; <c>embedding_norm.{weight,bias}</c> (LayerNorm of the summed embeddings);
/// per layer <c>layers.{i}.</c>: <c>attention.{q,k,v,o}.weight</c> (+ optional <c>.bias</c>), <c>attention_norm.{weight,bias}</c>
/// (LayerNorm of residual + attention output), <c>mlp.up.weight</c>, optional <c>mlp.gate.weight</c> (gated), <c>mlp.down.weight</c>
/// (+ optional biases), <c>mlp_norm.{weight,bias}</c> (LayerNorm of residual + MLP output). Optional classifier head on the first
/// row (cls): <c>classifier.pre.{weight,bias}</c> (dense + tanh; BERT pooler / RoBERTa head dense) and <c>classifier.out.{weight,bias}</c>
/// [labels, H].</para>
/// </remarks>
public sealed class EncoderAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Bidirectional transformer encoder";
    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["encoder"];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int nLayer = src.Arch<int>("layer_count");
        int nEmbd = src.Arch<int>("hidden_size");
        int nHead = src.Arch<int>("attention.head_count");
        int headDim = nEmbd / nHead;
        float eps = src.Arch("norm_eps", 1e-12f);
        string act = src.Arch("mlp.activation", "gelu");
        var posTable = src.Weight("position_embedding.weight");
        bool rope = posTable is null;
        int ropeDim = src.Arch("rope.dim", headDim);
        float ropeBase = src.Arch("rope.base", 10000f);
        string ropeStyle = src.Arch("rope.style", "neox");
        int ctx = src.Arch("context_length", posTable is null ? 2048 : (int)posTable.Shape[0].Fixed);
        if (posTable is not null)
        {
            ctx = (int)Math.Min(ctx, posTable.Shape[0].Fixed);
        }

        string pooling = PackedSequences.NormalizePooling(options.EmbeddingPooling) ?? PackedSequences.NormalizePooling(src.Get<string?>(CanonicalKeys.EmbeddingPooling, null)) ?? "cls";
        bool normalize = options.EmbeddingNormalize ?? src.Get(CanonicalKeys.EmbeddingNormalize, true);
        bool hasTypes = src.Weight("token_type_embedding.weight") is not null;
        bool hasClassifier = src.Weight("classifier.out.weight") is not null;
        int labels = hasClassifier ? (int)src.Weight("classifier.out.weight")!.Shape[0].Fixed : 0;

        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.EmbedGraph] = BuildGraph(WellKnown.EmbedGraph, classify: false) };
        if (hasClassifier)
        {
            graphs[WellKnown.ClassifyGraph] = BuildGraph(WellKnown.ClassifyGraph, classify: true);
        }

        var props = new Dictionary<string, object>
        {
            [WellKnown.VocabSize] = (int)src.Weight("token_embedding.weight")!.Shape[0].Fixed,
            [WellKnown.ContextLength] = ctx,
            [WellKnown.EmbeddingLength] = nEmbd,
            [WellKnown.PositionAxes] = 1,
            [WellKnown.EmbeddingPooling] = pooling,
            [WellKnown.EmbeddingNormalize] = normalize,
            [WellKnown.CausalAttention] = false,
            ["n_layer"] = nLayer,
        };
        if (hasClassifier)
        {
            props[WellKnown.ClassifierLabels] = labels;
        }

        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = graphs,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Properties = props,
        };

        ComputeGraph BuildGraph(string name, bool classify)
        {
            var b = new GraphBuilder(name, src.Weight);
            var inp = PackedSequences.Declare(b, hasTypes);
            Value x;
            using (b.Group("input"))
            {
                x = b.Gather(b.Weight("token_embedding.weight"), inp.Tokens, "inp_embd");
                if (posTable is not null)
                {
                    x = b.Add(x, b.Gather(b.Weight("position_embedding.weight"), b.Reshape(inp.Positions, 0), "pos_embd"));
                }

                if (hasTypes)
                {
                    x = b.Add(x, b.Gather(b.Weight("token_type_embedding.weight"), inp.TokenTypes!, "type_embd"));
                }

                x = LayerNorm(b, x, "embedding_norm");
            }
            for (int il = 0; il < nLayer; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"layers.{il}.";
                var q = b.Reshape(b.Linear(x, p + "attention.q.weight", p + "attention.q.bias"), 0, nHead, headDim);
                var k = b.Reshape(b.Linear(x, p + "attention.k.weight", p + "attention.k.bias"), 0, nHead, headDim);
                var v = b.Reshape(b.Linear(x, p + "attention.v.weight", p + "attention.v.bias"), 0, nHead, headDim);
                if (rope)
                {
                    var ra = Attributes.From(("dims", ropeDim), ("base", ropeBase), ("style", ropeStyle));
                    q = b.Emit1("Rope", [q, inp.Positions], ra);
                    k = b.Emit1("Rope", [k, inp.Positions], ra);
                }
                var attn = PackedSequences.Attention(b, q, k, v, inp.RowSpans, 1f / MathF.Sqrt(headDim), causal: false, name: $"{p}attn");
                var o = b.Linear(b.Reshape(attn, 0, nEmbd), p + "attention.o.weight", p + "attention.o.bias", $"{p}attn_out");
                x = LayerNorm(b, b.Add(x, o), p + "attention_norm");

                var up = b.Linear(x, p + "mlp.up.weight", p + "mlp.up.bias");
                var h = b.HasWeight(p + "mlp.gate.weight") ? b.Mul(up, Act(b, b.Linear(x, p + "mlp.gate.weight", p + "mlp.gate.bias"))) : Act(b, up);
                h = b.Linear(h, p + "mlp.down.weight", p + "mlp.down.bias", $"{p}ffn_out");
                x = LayerNorm(b, b.Add(x, h), p + "mlp_norm");
            }
            using (b.Group("output"))
            {
                if (!classify)
                {
                    PackedSequences.EmbeddingsOutput(b, x, inp.SequenceSpans, pooling, normalize);
                }
                else
                {
                    var cls = PackedSequences.Pool(b, x, inp.SequenceSpans, "cls", "cls_rows");
                    if (b.HasWeight("classifier.pre.weight"))
                    {
                        cls = b.Unary("Tanh", b.Linear(cls, "classifier.pre.weight", "classifier.pre.bias"));
                    }

                    b.Output(WellKnown.LogitsOutput, b.Linear(cls, "classifier.out.weight", "classifier.out.bias", "logits"));
                }
            }
            return b.Build();
        }

        Value LayerNorm(GraphBuilder b, Value v, string name)
            => b.Emit1("LayerNorm", [v, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", eps)));

        Value Act(GraphBuilder b, Value t) => act switch
        {
            "gelu" => b.Unary("Gelu", t),
            "gelu_tanh" => b.Unary("GeluTanh", t),
            "relu" => b.Unary("Relu", t),
            "silu" => b.Silu(t),
            _ => throw new NotSupportedException($"encoder: unknown mlp.activation '{act}'."),
        };
    }
}
