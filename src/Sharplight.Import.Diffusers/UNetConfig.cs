using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>Hyper-parameters of a latent-diffusion UNet, written as the canonical <c>latent_diffusion.unet.*</c> keys.</summary>
internal sealed record UNetConfig
{
    /// <summary>Latent channels in.</summary>
    public required int InChannels { get; init; }

    /// <summary>Channels predicted.</summary>
    public required int OutChannels { get; init; }

    /// <summary>Channels per resolution level, finest first.</summary>
    public required int[] BlockChannels { get; init; }

    /// <summary>Residual blocks per level on the way down (the way up has one more).</summary>
    public required int LayersPerBlock { get; init; }

    /// <summary>Transformer blocks of each level's spatial transformers on the way down, finest first (0: the level has no attention).</summary>
    public required int[] DownTransformerDepths { get; init; }

    /// <summary>Transformer blocks of each level's spatial transformers on the way up, coarsest first (0: no attention).</summary>
    public required int[] UpTransformerDepths { get; init; }

    /// <summary>Attention heads per level, finest first.</summary>
    public required int[] HeadCounts { get; init; }

    /// <summary>Transformer blocks of the middle block's spatial transformer (0: none).</summary>
    public required int MidTransformerDepth { get; init; }

    /// <summary>Width of the text conditioning (cross-attention keys and values).</summary>
    public required int ContextDim { get; init; }

    /// <summary>Latent height and width the model was trained at.</summary>
    public required int SampleSize { get; init; }

    /// <summary>GroupNorm groups.</summary>
    public int NormGroups { get; init; } = 32;

    /// <summary>GroupNorm epsilon of the residual blocks and the output norm.</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>Whether the sinusoidal timestep embedding puts the cosines first.</summary>
    public bool FlipSinToCos { get; init; } = true;

    /// <summary>Frequency shift of the sinusoidal timestep embedding.</summary>
    public float FreqShift { get; init; }

    /// <summary>Width of the extra conditioning vector added to the time embedding (SDXL: pooled text + size embeddings), 0 for none.</summary>
    public int ClassEmbeddingInput { get; init; }

    /// <summary>Width of each sinusoidal size embedding of the extra conditioning (SDXL: 256), 0 for none.</summary>
    public int AdditionTimeDim { get; init; }

    /// <summary>Writes the keys under <c>{prefix}unet.</c>.</summary>
    /// <param name="metadata">The canonical metadata.</param>
    /// <param name="prefix">The architecture prefix (<c>latent_diffusion.</c>).</param>
    public void WriteTo(IDictionary<string, object> metadata, string prefix)
    {
        string p = prefix + "unet.";
        metadata[p + "in_channels"] = InChannels;
        metadata[p + "out_channels"] = OutChannels;
        metadata[p + "block_channels"] = BlockChannels;
        metadata[p + "layers_per_block"] = LayersPerBlock;
        metadata[p + "down_transformer_depths"] = DownTransformerDepths;
        metadata[p + "up_transformer_depths"] = UpTransformerDepths;
        metadata[p + "head_counts"] = HeadCounts;
        metadata[p + "mid_transformer_depth"] = MidTransformerDepth;
        metadata[p + "context_dim"] = ContextDim;
        metadata[p + "sample_size"] = SampleSize;
        metadata[p + "norm_groups"] = NormGroups;
        metadata[p + "norm_eps"] = NormEps;
        metadata[p + "flip_sin_to_cos"] = FlipSinToCos;
        metadata[p + "freq_shift"] = FreqShift;
        metadata[p + "class_embedding_input"] = ClassEmbeddingInput;
        metadata[p + "addition_time_dim"] = AdditionTimeDim;
    }

    /// <summary>Reads a diffusers <c>unet/config.json</c>.</summary>
    /// <param name="config">The parsed configuration.</param>
    /// <exception cref="NotSupportedException">The UNet uses features the graphs do not implement.</exception>
    public static UNetConfig FromDiffusers(JsonElement config)
    {
        int[] channels = config.Ints("block_out_channels");
        int levels = channels.Length;
        string[] downTypes = config.Strings("down_block_types");
        string[] upTypes = config.Strings("up_block_types");
        int[] perLevel = config.IntOrList("transformer_layers_per_block", 1, levels);
        int[] heads = config.TryGetProperty("num_attention_heads", out var explicitHeads) && explicitHeads.ValueKind != JsonValueKind.Null
            ? config.IntOrList("num_attention_heads", 8, levels)
            : config.IntOrList("attention_head_dim", 8, levels);
        int[] reversedDepths = config.TryGetProperty("reverse_transformer_layers_per_block", out var reverse) && reverse.ValueKind == JsonValueKind.Array
            ? [.. reverse.EnumerateArray().Select(e => e.EnumerateArray().First().GetInt32())]
            : [.. perLevel.Reverse()];
        string time = config.String("time_embedding_type", "positional");
        if (time != "positional" || config.String("resnet_time_scale_shift", "default") != "default")
        {
            throw new NotSupportedException($"UNet time embedding '{time}' / scale-shift residual blocks are not supported.");
        }

        string addition = config.String("addition_embed_type", "");
        int additionTime = addition == "text_time" ? config.Int("addition_time_embed_dim", 256) : 0;
        return new UNetConfig
        {
            InChannels = config.Int("in_channels", 4),
            OutChannels = config.Int("out_channels", 4),
            BlockChannels = channels,
            LayersPerBlock = config.Int("layers_per_block", 2),
            DownTransformerDepths = [.. downTypes.Select((type, level) => type.StartsWith("CrossAttn", StringComparison.Ordinal) ? perLevel[level] : 0)],
            UpTransformerDepths = [.. upTypes.Select((type, level) => type.StartsWith("CrossAttn", StringComparison.Ordinal) ? reversedDepths[level] : 0)],
            HeadCounts = heads,
            MidTransformerDepth = config.String("mid_block_type", "UNetMidBlock2DCrossAttn") == "UNetMidBlock2DCrossAttn" ? perLevel[^1] : 0,
            ContextDim = config.Int("cross_attention_dim", 768),
            SampleSize = config.Int("sample_size", 64),
            NormGroups = config.Int("norm_num_groups", 32),
            NormEps = config.Float("norm_eps", 1e-5f),
            FlipSinToCos = config.Bool("flip_sin_to_cos", true),
            FreqShift = config.Float("freq_shift", 0),
            ClassEmbeddingInput = additionTime > 0 ? config.Int("projection_class_embeddings_input_dim", 0) : 0,
            AdditionTimeDim = additionTime,
        };
    }

    /// <summary>
    /// Infers the configuration of a UNet from its canonical weights (single-file and GGUF checkpoints carry no configuration):
    /// levels and widths from the residual blocks, transformer depths from the blocks present, heads from the context width
    /// (SD 1.x: 8 heads per level; SD 2.x and SDXL: 64-wide heads).
    /// </summary>
    /// <param name="weights">The canonical weights (<c>unet.*</c>).</param>
    /// <param name="sampleSize">The latent size the family was trained at.</param>
    public static UNetConfig FromWeights(IReadOnlyDictionary<string, WeightInfo> weights, int sampleSize)
    {
        int levels = Count(weights, "unet.down.");
        int layers = Count(weights, "unet.down.0.res.");
        int[] channels = [.. Enumerable.Range(0, levels).Select(l => (int)weights[$"unet.down.{l}.res.0.conv1.weight"].Shape[0].Fixed)];
        int[] down = [.. Enumerable.Range(0, levels).Select(l => Count(weights, $"unet.down.{l}.attn.0.blocks."))];
        int[] up = [.. Enumerable.Range(0, levels).Select(l => Count(weights, $"unet.up.{l}.attn.0.blocks."))];
        int mid = Count(weights, "unet.mid.attn.0.blocks.");
        var crossKey = weights.Keys.First(k => k.EndsWith(".cross.k.weight", StringComparison.Ordinal));
        int context = (int)weights[crossKey].Shape[1].Fixed;
        int[] heads = [.. channels.Select(c => context == 768 ? 8 : Math.Max(1, c / 64))];
        bool sdxl = weights.ContainsKey("unet.class_embedding.linear1.weight");
        return new UNetConfig
        {
            InChannels = (int)weights["unet.input_conv.weight"].Shape[1].Fixed,
            OutChannels = (int)weights["unet.output_conv.weight"].Shape[0].Fixed,
            BlockChannels = channels,
            LayersPerBlock = layers,
            DownTransformerDepths = down,
            UpTransformerDepths = up,
            HeadCounts = heads,
            MidTransformerDepth = mid,
            ContextDim = context,
            SampleSize = sampleSize,
            ClassEmbeddingInput = sdxl ? (int)weights["unet.class_embedding.linear1.weight"].Shape[1].Fixed : 0,
            AdditionTimeDim = sdxl ? 256 : 0,
        };
    }

    /// <summary>Number of distinct indices directly under <paramref name="prefix"/> (<c>prefix{i}.</c>).</summary>
    private static int Count(IReadOnlyDictionary<string, WeightInfo> weights, string prefix)
    {
        var indices = new HashSet<int>();
        foreach (string name in weights.Keys)
        {
            if (!name.StartsWith(prefix, StringComparison.Ordinal))
            {
                continue;
            }

            int end = name.IndexOf('.', prefix.Length);
            if (end > prefix.Length && int.TryParse(name.AsSpan(prefix.Length, end - prefix.Length), out int index))
            {
                indices.Add(index);
            }
        }

        return indices.Count;
    }
}
