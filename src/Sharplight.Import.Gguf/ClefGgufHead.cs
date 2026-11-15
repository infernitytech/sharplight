using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Gguf;

/// <summary>
/// The decision head of llama.cpp <c>clef</c> (Cloudflare Clef: the <c>qwen35</c> trunk plus a joint schema head) → the canonical
/// "joint_schema" span scoring head: <c>decision.*</c> and <c>token_types.weight</c> are global tensors, the head blocks
/// <c>dec.blk.N.*</c> are the routing layers first (<c>decision.routing_block_count</c>), then the joint layers (<c>decision.block_count</c>).
/// </summary>
internal static partial class ClefGgufHead
{
    /// <summary>The GGUF architecture.</summary>
    public const string Architecture = "clef";

    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["decision.hidden_norm.weight"] = "hidden_norm.weight",
        ["decision.hidden_norm.bias"] = "hidden_norm.bias",
        ["decision.proj_memory.weight"] = "memory.weight",
        ["decision.proj_question.weight"] = "question.weight",
        ["decision.proj_option_question.weight"] = "option_question.weight",
        ["decision.proj_global.weight"] = "global.weight",
        ["decision.proj_option_context.weight"] = "option_context.weight",
        ["decision.proj_option_lexical.weight"] = "option_lexical.weight",
        ["decision.option_summary_norm.weight"] = "option_summary_norm.weight",
        ["decision.option_summary_norm.bias"] = "option_summary_norm.bias",
        ["decision.field_norm.weight"] = "field_norm.weight",
        ["decision.field_norm.bias"] = "field_norm.bias",
        ["decision.option_norm.weight"] = "option_norm.weight",
        ["decision.option_norm.bias"] = "option_norm.bias",
        ["decision.scales"] = "scales",
        ["decision.scorer.weight"] = "scorer.weight",
        ["decision.scorer.bias"] = "scorer.bias",
        ["decision.scorer_out.weight"] = "scorer_out.weight",
        ["decision.scorer_out.bias"] = "scorer_out.bias",
        ["token_types.weight"] = "type_embedding.weight",
    };

    // Per block: the routing layer's name, the joint layer's name (null: the block kind has no such tensor).
    private static readonly Dictionary<string, (string? Routing, string? Joint)> s_block = new(StringComparer.Ordinal)
    {
        ["cross_attn_norm_kv"] = ("memory_norm", null),
        ["cross_attn_norm"] = ("query_norm", "cross_norm"),
        ["cross_attn_q"] = ("attention.q", "cross_attention.q"),
        ["cross_attn_k"] = ("attention.k", "cross_attention.k"),
        ["cross_attn_v"] = ("attention.v", "cross_attention.v"),
        ["cross_attn_o"] = ("attention.o", "cross_attention.o"),
        ["attn_norm"] = (null, "self_norm"),
        ["attn_q"] = (null, "self_attention.q"),
        ["attn_k"] = (null, "self_attention.k"),
        ["attn_v"] = (null, "self_attention.v"),
        ["attn_o"] = (null, "self_attention.o"),
        ["ffn_norm"] = ("ffn_norm", "ffn_norm"),
        ["ffn_up"] = ("ffn_up", "ffn_up"),
        ["ffn_down"] = ("ffn_down", "ffn_down"),
    };

    [GeneratedRegex(@"^dec\.blk\.(\d+)\.([a-z_]+)\.(weight|bias)$")]
    private static partial Regex Block();

    /// <summary>Adds the head's canonical hyper-parameters (<c>span_head.*</c>).</summary>
    public static void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[CanonicalKeys.SpanHeadType] = "joint_schema";
        md[CanonicalKeys.SpanHeadNormEps] = g.Float("attention.layer_norm_epsilon", 1e-5f);
        md[CanonicalKeys.SpanHeadHeadCount] = g.Int("decision.head_count");
        md[CanonicalKeys.SpanHeadRoutingLayerCount] = g.Int("decision.routing_block_count");
        md[CanonicalKeys.SpanHeadLayerCount] = g.Int("decision.block_count");
    }

    /// <summary>The canonical name of a head tensor, or null when <paramref name="name"/> is not one.</summary>
    public static string? MapWeightName(string name, GgufMetadataView g)
    {
        if (s_global.TryGetValue(name, out var global))
        {
            return "span_head." + global;
        }

        var m = Block().Match(name);
        if (!m.Success || !s_block.TryGetValue(m.Groups[2].Value, out var names))
        {
            return null;
        }

        int block = int.Parse(m.Groups[1].Value);
        int routing = g.Int("decision.routing_block_count");
        string? local = block < routing ? names.Routing : names.Joint;
        return local is null ? null
            : block < routing ? $"span_head.routing.{block}.{local}.{m.Groups[3].Value}" : $"span_head.layers.{block - routing}.{local}.{m.Groups[3].Value}";
    }
}
