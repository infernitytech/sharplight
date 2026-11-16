using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Token-activated adapter routing (Granite Switch): activation tokens select an adapter for themselves and every later token of their
/// sequence. The table <see cref="TokenTable"/> [V, 8] holds, per token, the token it is embedded as (substitute id, as a float), its router key
/// (+gain for an activation token, −gain otherwise) and its router value (the adapter's slot, 0 otherwise). A single causal attention head
/// with query 1 over these keys and values (a KV cache of its own) averages the slots of the activation tokens seen so far, as the other tokens'
/// weights vanish; the result is clamped to [0, adapters] and rounded half up to the row's slot (0 = no adapter, see <see cref="SwitchedLora"/>).
/// </summary>
internal static class AdapterRouter
{
    /// <summary>Canonical name of the per-token table [V, 8] (substitute token, router key, router value, zero padding).</summary>
    public const string TokenTable = "adapter_tokens.weight";

    /// <summary>Head size of the router's attention (only the first channel carries a signal).</summary>
    public const int HeadDim = 32;

    /// <summary>Whether the model routes adapters by token.</summary>
    public static bool IsPresent(ModelSource source) => source.Weight(TokenTable) is not null;

    /// <summary>The router's key and value caches (one head of <see cref="HeadDim"/>).</summary>
    /// <param name="group">State group (and name prefix) of the caches.</param>
    /// <param name="type">Cache element type.</param>
    public static (StateDefinition Key, StateDefinition Value) CreateCaches(string group, DataType type)
    {
        var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), 1, HeadDim);
        return (new StateDefinition { Name = $"{group}.k_cache", Type = type, Shape = shape, Kind = StateKind.KeyCache, Group = group },
                new StateDefinition { Name = $"{group}.v_cache", Type = type, Shape = shape, Kind = StateKind.ValueCache, Group = group });
    }

    /// <summary>Routes the rows of <paramref name="tokens"/>.</summary>
    /// <param name="b">Graph under construction.</param>
    /// <param name="tokens">Token ids, i32 [T].</param>
    /// <param name="adapters">Number of adapters (the largest slot).</param>
    /// <param name="attend">Causal attention of queries over keys and values [T, 1, <see cref="HeadDim"/>] (cached or within packed sequences),
    /// returning [T, 1, <see cref="HeadDim"/>]; it applies no scale.</param>
    /// <returns>The tokens to embed (activation tokens replaced by their substitutes), i32 [T], and every row's adapter slot, i32 [T, 1].</returns>
    public static (Value Tokens, Value Slots) Route(GraphBuilder b, Value tokens, int adapters, Func<Value, Value, Value, Value> attend)
    {
        using var _ = b.Group("adapter_router");
        var rows = b.Gather(b.Weight(TokenTable), tokens, "adapter_token_rows");                                       // [T, 8]
        var substituted = b.Reshape(b.Emit1("Cast", [b.Slice(rows, 1, 0, 1)], Attributes.From(("type", "i32"))), 0);   // [T]
        var firstChannel = new byte[HeadDim * sizeof(float)];
        BitConverter.TryWriteBytes(firstChannel, 1f);
        var channel = b.Constant("adapter_router_channel", DataType.F32, new Shape(1, HeadDim), firstChannel);
        var key = b.Slice(rows, 1, 1, 1);                                                                             // [T, 1]
        Value Lane(Value column) => b.Reshape(b.Mul(column, channel), 0, 1, HeadDim);                                  // [T, 1, D]
        var routed = attend(Lane(b.Scale(key, 0f, 1f)), Lane(key), Lane(b.Slice(rows, 1, 2, 1)));                     // [T, 1, D]
        var slot = b.Slice(b.Reshape(routed, 0, HeadDim), 1, 0, 1);                                                   // [T, 1]
        var clamped = b.Emit1("Clip", [slot], Attributes.From(("min", 0f), ("max", (float)adapters)));
        var rounded = b.Unary("Floor", b.Scale(clamped, 1f, 0.5f));
        return (substituted, b.Emit1("Cast", [rounded], Attributes.From(("type", "i32")), "adapter_slots"));
    }
}
