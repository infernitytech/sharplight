using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a DFlash drafter (<c>dflash.*</c>, see <see cref="DflashDraftHead"/>), checked against its target.</summary>
internal sealed class DflashParameters
{
    public DflashParameters(ModelSource drafter, int targetHidden)
    {
        Hidden = drafter.Arch<int>("hidden_size");
        Layers = drafter.Arch<int>("layer_count");
        Heads = drafter.Arch<int>("head_count");
        KvHeads = drafter.Arch<int>("kv_head_count");
        HeadDim = drafter.Arch<int>("head_dim");
        Eps = drafter.Arch("norm_eps", 1e-6f);
        RopeBase = drafter.Arch("rope.base", 10000f);
        RopeDims = drafter.Arch("rope.dims", HeadDim);
        RopeAttentionScale = drafter.Arch("rope.attention_scale", 1f);
        Features = DflashDraftHead.TargetLayers(drafter).Length * targetHidden;
        Causal = drafter.Arch("causal", false);
        Window = drafter.Arch("sliding_window", 0);
        var sliding = drafter.Arch("sliding_layers", Array.Empty<int>());
        SlidingLayers = [.. Enumerable.Range(0, Layers).Select(il => Window > 0 && (sliding.Length == 0 || (il < sliding.Length && sliding[il] != 0)))];
        ConvTaps = DflashDraftHead.ConvTaps(drafter);
        ConvGroupSize = drafter.Arch("conv.group_size", 1);
        LatticeCandidates = DflashDraftHead.LatticeCandidates(drafter);
        SelectorRank = drafter.Arch("selector.rank", 0);
        EmbeddingScale = drafter.Arch("embedding_scale", 0f);
        AttentionScale = drafter.Arch("attention.scale", 0f) is var scale && scale != 0f ? scale : 1f / MathF.Sqrt(HeadDim);
        ValueScale = drafter.Arch("attention.value_scale", 0f);
        LogitScale = drafter.Arch("logit_scale", 0f);
        Softcap = drafter.Arch("final_logit_softcap", 0f);
        Activation = drafter.Arch("mlp.activation", "silu") switch
        {
            "silu" => "Silu",
            "gelu_tanh" => "GeluTanh",
            var other => throw new InvalidDataException($"dflash: MLP activation '{other}' is not supported."),
        };

        if (drafter.Weight("fc.weight") is { } fc && fc.Shape[-1] != Features)
        {
            throw new InvalidDataException($"dflash: the drafter fuses {fc.Shape[-1]} feature columns, the target's {DflashDraftHead.TargetLayers(drafter).Length} layers have {Features}.");
        }

        if (ConvTaps > 0 && (ConvGroupSize <= 0 || Hidden % ConvGroupSize != 0))
        {
            throw new InvalidDataException($"dflash: conv.group_size {ConvGroupSize} does not divide the hidden size {Hidden}.");
        }

        if (LatticeCandidates > 0 && SelectorRank <= 0)
        {
            throw new InvalidDataException("dflash: the selector has no rank.");
        }
    }

    public int Hidden { get; }
    public int Layers { get; }
    public int Heads { get; }
    public int KvHeads { get; }
    public int HeadDim { get; }
    public float Eps { get; }
    public float RopeBase { get; }
    public int RopeDims { get; }
    public float RopeAttentionScale { get; }
    /// <summary>Width of the target features (taps · target hidden size).</summary>
    public int Features { get; }
    public bool Causal { get; }
    public int Window { get; }
    public bool[] SlidingLayers { get; }
    public int ConvTaps { get; }
    public int ConvGroupSize { get; }
    public int LatticeCandidates { get; }
    public int SelectorRank { get; }
    public float EmbeddingScale { get; }
    public float AttentionScale { get; }
    public float ValueScale { get; }
    public float LogitScale { get; }
    public float Softcap { get; }
    /// <summary>The unary op of the gated MLP.</summary>
    public string Activation { get; }

    /// <summary>Width of the hidden-state input rows: features, context flag, two columns per convolution tap, two for the lattice link.</summary>
    public int InputWidth => Features + 1 + (2 * ConvTaps) + (LatticeCandidates > 0 ? 2 : 0);
}
