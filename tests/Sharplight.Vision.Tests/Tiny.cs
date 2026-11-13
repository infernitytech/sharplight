using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vision.Tests;

/// <summary>A tiny, randomly initialised Qwen3.5 text model (canonical vocabulary) plus a tiny vision tower.</summary>
internal static class Tiny
{
    public const string Arch = "qwen3_5";
    public const int Vocab = 40, Hidden = 32;
    public static readonly VisionParameters Vision = new()
    {
        LayerCount = 2, HiddenSize = 16, FfnHiddenSize = 32, HeadCount = 2, NormEps = 1e-6f, Activation = "gelu_tanh",
        PatchSize = 4, TemporalPatchSize = 2, InChannels = 3, SpatialMergeSize = 2, PositionEmbeddingCount = 16, RopeBase = 10000f,
        OutHiddenSize = Hidden, MergerActivation = "gelu", ImageMean = [0.5f, 0.4f, 0.3f], ImageStd = [0.5f, 0.6f, 0.7f],
        MinPixels = 16 * 16 * 4, MaxPixels = 64 * 64 * 4,
    };

    public static WeightInfo Weight(string name, Random rng, float scale, params long[] dims)
    {
        long n = dims.Aggregate(1L, (a, b) => a * b);
        var f = new float[n];
        for (int i = 0; i < n; i++)
        {
            f[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        }

        return new WeightInfo { Name = name, Type = DataType.F32, Shape = new Shape(dims.Select(d => (Dim)d)), Source = new MemSource(MemoryMarshal.AsBytes(f.AsSpan()).ToArray()) };
    }

    public static WeightInfo Ones(string name, params long[] dims)
    {
        long n = dims.Aggregate(1L, (a, b) => a * b);
        var f = Enumerable.Repeat(1f, (int)n).ToArray();
        return new WeightInfo { Name = name, Type = DataType.F32, Shape = new Shape(dims.Select(d => (Dim)d)), Source = new MemSource(MemoryMarshal.AsBytes(f.AsSpan()).ToArray()) };
    }

    /// <summary>Canonical vision.* weights for <see cref="Vision"/>.</summary>
    public static Dictionary<string, WeightInfo> VisionWeights(int seed = 3)
    {
        var p = Vision; var rng = new Random(seed); var w = new Dictionary<string, WeightInfo>();
        long H = p.HiddenSize, F = p.FfnHiddenSize, M = p.MergeUnit * H;
        void Add(WeightInfo x) => w[x.Name] = x;
        void Norm(string n) { Add(Ones(n + ".weight", H)); Add(Weight(n + ".bias", rng, 0.1f, H)); }
        Add(Weight("vision.patch_embed.weight", rng, 0.2f, H, p.PatchDim)); Add(Weight("vision.patch_embed.bias", rng, 0.1f, H));
        Add(Weight("vision.position_embedding.weight", rng, 0.5f, p.PositionEmbeddingCount, H));
        for (int i = 0; i < p.LayerCount; i++)
        {
            string l = $"vision.layers.{i}.";
            Norm(l + "input_norm"); Norm(l + "post_attention_norm");
            Add(Weight(l + "attention.qkv.weight", rng, 0.3f, 3 * H, H)); Add(Weight(l + "attention.qkv.bias", rng, 0.1f, 3 * H));
            Add(Weight(l + "attention.o.weight", rng, 0.3f, H, H)); Add(Weight(l + "attention.o.bias", rng, 0.1f, H));
            Add(Weight(l + "mlp.up.weight", rng, 0.3f, F, H)); Add(Weight(l + "mlp.up.bias", rng, 0.1f, F));
            Add(Weight(l + "mlp.down.weight", rng, 0.3f, H, F)); Add(Weight(l + "mlp.down.bias", rng, 0.1f, H));
        }
        Norm("vision.merger.norm");
        Add(Weight("vision.merger.up.weight", rng, 0.2f, M, M)); Add(Weight("vision.merger.up.bias", rng, 0.1f, M));
        Add(Weight("vision.merger.down.weight", rng, 0.2f, p.OutHiddenSize, M)); Add(Weight("vision.merger.down.bias", rng, 0.1f, p.OutHiddenSize));
        return w;
    }

    public static Dictionary<string, object> VisionMetadata(string arch = Arch)
    {
        var props = new Dictionary<string, object>();
        Vision.WriteProperties(props);
        return props.ToDictionary(kv => $"{arch}.{kv.Key}", kv => kv.Value);
    }

    /// <summary>Text trunk: 2 layers (linear attention, then full attention), canonical names.</summary>
    public static ModelSource Text(bool withVision, int seed = 1)
    {
        var rng = new Random(seed);
        var w = new Dictionary<string, WeightInfo>();
        void Add(WeightInfo x) => w[x.Name] = x;
        const long H = Hidden, FF = 64, heads = 2, kv = 1, D = 16, hk = 1, dk = 8, hv = 2, dv = 8;
        Add(Weight("token_embedding.weight", rng, 0.5f, Vocab, H));
        Add(Ones("output_norm.weight", H));
        for (int il = 0; il < 2; il++)
        {
            string l = $"layers.{il}.";
            Add(Ones(l + "input_norm.weight", H)); Add(Ones(l + "post_attention_norm.weight", H));
            Add(Weight(l + "mlp.gate.weight", rng, 0.2f, FF, H)); Add(Weight(l + "mlp.up.weight", rng, 0.2f, FF, H)); Add(Weight(l + "mlp.down.weight", rng, 0.2f, H, FF));
            if (il == 0)
            {
                Add(Weight(l + "linear_attention.qkv.weight", rng, 0.2f, 2 * hk * dk + hv * dv, H));
                Add(Weight(l + "linear_attention.z.weight", rng, 0.2f, hv * dv, H));
                Add(Weight(l + "linear_attention.beta.weight", rng, 0.2f, hv, H));
                Add(Weight(l + "linear_attention.alpha.weight", rng, 0.2f, hv, H));
                Add(Weight(l + "linear_attention.out.weight", rng, 0.2f, H, hv * dv));
                Add(Ones(l + "linear_attention.norm.weight", dv));
                Add(Weight(l + "linear_attention.conv1d.weight", rng, 0.3f, 2 * hk * dk + hv * dv, 4));
                Add(Weight(l + "linear_attention.a", rng, 0.5f, hv)); Add(Weight(l + "linear_attention.dt_bias", rng, 0.5f, hv));
            }
            else
            {
                Add(Weight(l + "attention.q.weight", rng, 0.2f, heads * 2 * D, H));
                Add(Weight(l + "attention.k.weight", rng, 0.2f, kv * D, H));
                Add(Weight(l + "attention.v.weight", rng, 0.2f, kv * D, H));
                Add(Weight(l + "attention.o.weight", rng, 0.2f, H, heads * D));
                Add(Ones(l + "attention.q_norm.weight", D)); Add(Ones(l + "attention.k_norm.weight", D));
            }
        }
        var md = new Dictionary<string, object>
        {
            [CanonicalKeys.Architecture] = Arch, [CanonicalKeys.Name] = "tiny",
            [$"{Arch}.layer_count"] = 2, [$"{Arch}.hidden_size"] = (int)H, [$"{Arch}.ffn_hidden_size"] = (int)FF, [$"{Arch}.context_length"] = 256,
            [$"{Arch}.norm_eps"] = 1e-6f, [$"{Arch}.attention.head_count"] = (int)heads, [$"{Arch}.attention.kv_head_count"] = (int)kv,
            [$"{Arch}.attention.head_dim"] = (int)D, [$"{Arch}.rope.dim"] = 8, [$"{Arch}.rope.base"] = 10000f, [$"{Arch}.rope.sections"] = new[] { 2, 1, 1, 0 },
            [$"{Arch}.full_attention_interval"] = 2, [$"{Arch}.linear_attention.conv_kernel"] = 4,
            [$"{Arch}.linear_attention.key_head_count"] = (int)hk, [$"{Arch}.linear_attention.key_head_dim"] = (int)dk,
            [$"{Arch}.linear_attention.value_head_count"] = (int)hv, [$"{Arch}.linear_attention.value_head_dim"] = (int)dv,
        };
        if (withVision)
        {
            foreach (var (k, v) in VisionMetadata())
            {
                md[k] = v;
            }

            foreach (var (k, v) in VisionWeights())
            {
                w[k] = v;
            }
        }
        return new ModelSource { Format = "test", Architecture = Arch, Metadata = md, Weights = w };
    }

    /// <summary>A vision-only auxiliary source (as an "mmproj" file would give).</summary>
    public static ModelSource VisionOnly() => new()
    {
        Format = "test", Architecture = Arch, Metadata = VisionMetadata(), Weights = VisionWeights(),
    };

    public static ModelDefinition Build(bool visionInPrimary, bool visionAuxiliary)
    {
        var primary = Text(visionInPrimary);
        var aux = visionAuxiliary ? new Dictionary<string, ModelSource> { [AuxiliaryRoles.Mmproj] = VisionOnly() } : [];
        return new Sharplight.Architectures.Qwen35Adapter().Build(new ModelSourceSet { Primary = primary, Auxiliary = aux }, new AdapterOptions());
    }
}
