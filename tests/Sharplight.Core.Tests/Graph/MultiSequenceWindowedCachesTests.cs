using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Core.Tests.Graph;

/// <summary>
/// <see cref="MultiSequence.WindowedCaches"/>: the raw keys of a compressed attention (DeepSeek-V4's heavily compressed layers: a window of raw
/// keys plus compressed entries, no selection) stay full-length caches. The op takes one block table for both caches, and a window block
/// table recycles the blocks the window has passed, which the compressed entries are still addressed through.
/// </summary>
public sealed class MultiSequenceWindowedCachesTests
{
    private const int Heads = 2;
    private const int Dims = 8;
    private const int Window = 4;

    [Fact]
    public void CompressedAttentionKeys_AreNotWindowed()
    {
        var b = new GraphBuilder("t", _ => null);
        var t = Dim.Sym("T");
        var q = b.Input("q", InputRole.Other, DataType.F32, new Shape(t, Heads, Dims));
        var k = b.Input("k", InputRole.Other, DataType.F32, new Shape(t, 1, Dims));
        var ctx = Dim.Sym(WellKnown.ContextSymbol);
        var sinks = b.Constant("sinks", DataType.F32, new Shape(Heads), MemoryMarshal.AsBytes(new float[Heads].AsSpan()).ToArray());
        b.RuntimeScalar(WellKnown.PastScalar);

        var plainKeys = b.State(new StateDefinition { Name = "plain_keys", Type = DataType.F16, Kind = StateKind.KeyCache, Shape = new Shape(ctx, 1, Dims) });
        b.CacheWrite(plainKeys, k);
        var plain = Attributes.From(("scale", 0.5f), ("offset", WellKnown.PastScalar), ("window", (long)Window), ("sinks", true));
        b.Output("plain", b.Emit1("CachedAttention", [q, plainKeys, plainKeys, sinks], plain));

        var keys = b.State(new StateDefinition { Name = "keys", Type = DataType.F16, Kind = StateKind.KeyCache, Shape = new Shape(ctx, 1, Dims) });
        b.CacheWrite(keys, k);
        var entries = b.State(new StateDefinition
        {
            Name = "entries", Type = DataType.F16, Kind = StateKind.KeyCache, Compression = 4, Shape = new Shape(ctx.CeilDiv(4), 1, Dims),
        });
        b.Emit("CacheWrite", [entries, k], Attributes.From(("offset", WellKnown.PastScalar), ("compression", 4L)), writes: [entries]);
        b.Output("compressed", b.Emit1("CachedAttention", [q, keys, keys, sinks, entries], plain.With("compression", 4L)));

        var (window, caches) = MultiSequence.WindowedCaches(b.Build());
        Assert.Equal(Window, window);
        Assert.Equal(["plain_keys"], caches.Select(s => s.Name));
    }
}
