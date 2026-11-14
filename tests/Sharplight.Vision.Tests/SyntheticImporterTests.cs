using System.Text;
using System.Text.Json;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Hermetic importer tests: the same random tower is written as a Hugging Face checkpoint (config.json,
/// preprocessor_config.json, model.visual.* safetensors) and as a llama.cpp mmproj GGUF; both importers must yield the
/// same canonical tower, and equal to the tensors that went in.
/// </summary>
public class SyntheticImporterTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "slvision-imp-" + Guid.NewGuid().ToString("N")[..8]);
    private readonly List<IDisposable> _open = [];
    public SyntheticImporterTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { foreach (var d in _open) { d.Dispose(); } try { Directory.Delete(_dir, true); } catch { } }

    private static readonly VisionParameters P = Tiny.Vision;
    private static readonly float[] Mean = [0.48f, 0.46f, 0.41f];
    private static readonly float[] Std = [0.27f, 0.26f, 0.28f];

    /// <summary>HF tensor name → (shape, values) for the tower.</summary>
    private static Dictionary<string, (long[] Shape, float[] Data)> HfTensors()
    {
        var rng = new Random(11);
        var t = new Dictionary<string, (long[], float[])>();
        void Add(string n, params long[] shape)
        {
            var d = new float[shape.Aggregate(1L, (a, b) => a * b)];
            for (int i = 0; i < d.Length; i++) { d[i] = (float)(rng.NextDouble() * 2 - 1); }
            t["model.visual." + n] = (shape, d);
        }
        long H = P.HiddenSize;
        long F = P.FfnHiddenSize;
        long M = P.MergeUnit * H;
        Add("patch_embed.proj.weight", H, P.InChannels, P.TemporalPatchSize, P.PatchSize, P.PatchSize);
        Add("patch_embed.proj.bias", H);
        Add("pos_embed.weight", P.PositionEmbeddingCount, H);
        for (int i = 0; i < P.LayerCount; i++)
        {
            string b = $"blocks.{i}.";
            Add(b + "norm1.weight", H);
            Add(b + "norm1.bias", H);
            Add(b + "norm2.weight", H);
            Add(b + "norm2.bias", H);
            Add(b + "attn.qkv.weight", 3 * H, H);
            Add(b + "attn.qkv.bias", 3 * H);
            Add(b + "attn.proj.weight", H, H);
            Add(b + "attn.proj.bias", H);
            Add(b + "mlp.linear_fc1.weight", F, H);
            Add(b + "mlp.linear_fc1.bias", F);
            Add(b + "mlp.linear_fc2.weight", H, F);
            Add(b + "mlp.linear_fc2.bias", H);
        }
        Add("merger.norm.weight", H);
        Add("merger.norm.bias", H);
        Add("merger.linear_fc1.weight", M, M);
        Add("merger.linear_fc1.bias", M);
        Add("merger.linear_fc2.weight", P.OutHiddenSize, M);
        Add("merger.linear_fc2.bias", P.OutHiddenSize);
        return t;
    }

    private static byte[] Bytes(float[] d)
    {
        var b = new byte[d.Length * 4];
        Buffer.BlockCopy(d, 0, b, 0, b.Length);
        return b;
    }

    private string WriteHf(Dictionary<string, (long[] Shape, float[] Data)> tensors)
    {
        string dir = Path.Combine(_dir, "hf");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "config.json"), JsonSerializer.Serialize(new
        {
            model_type = "qwen3_5", architectures = new[] { "Qwen3_5ForConditionalGeneration" }, tie_word_embeddings = true,
            text_config = new
            {
                model_type = "qwen3_5_text", num_hidden_layers = 2, hidden_size = Tiny.Hidden, intermediate_size = 64, num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16,
                rms_norm_eps = 1e-6, full_attention_interval = 2, linear_conv_kernel_dim = 4, linear_key_head_dim = 8, linear_num_key_heads = 1, linear_value_head_dim = 8, linear_num_value_heads = 2,
                rope_parameters = new { rope_theta = 10000.0, partial_rotary_factor = 0.5, mrope_section = new[] { 2, 1, 1 } },
            },
            vision_config = new
            {
                depth = P.LayerCount, hidden_size = P.HiddenSize, intermediate_size = P.FfnHiddenSize, num_heads = P.HeadCount, patch_size = P.PatchSize, temporal_patch_size = P.TemporalPatchSize,
                spatial_merge_size = P.SpatialMergeSize, in_channels = P.InChannels, num_position_embeddings = P.PositionEmbeddingCount, out_hidden_size = P.OutHiddenSize,
                hidden_act = "gelu_pytorch_tanh", deepstack_visual_indexes = Array.Empty<int>(),
            },
        }));
        File.WriteAllText(Path.Combine(dir, "preprocessor_config.json"), JsonSerializer.Serialize(new
        {
            size = new { shortest_edge = P.MinPixels, longest_edge = P.MaxPixels }, image_mean = Mean, image_std = Std, patch_size = P.PatchSize,
        }));
        // safetensors: u64 header length, JSON header, data.
        var header = new Dictionary<string, object>();
        long off = 0;
        var data = new MemoryStream();
        foreach (var (name, (shape, d)) in tensors.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            var b = Bytes(d);
            header[name] = new { dtype = "F32", shape, data_offsets = new[] { off, off + b.Length } };
            data.Write(b);
            off += b.Length;
        }
        var h = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        using (var f = File.Create(Path.Combine(dir, "model.safetensors")))
        {
            f.Write(BitConverter.GetBytes((ulong)h.Length));
            f.Write(h);
            data.Position = 0;
            data.CopyTo(f);
        }
        return dir;
    }

    private string WriteMmproj(Dictionary<string, (long[] Shape, float[] Data)> hf)
    {
        string path = Path.Combine(_dir, "mmproj.gguf");
        var w = new GgufWriter()
            .AddMetadata("general.architecture", "clip").AddMetadata("general.type", "mmproj")
            .AddMetadata("clip.projector_type", "qwen3vl_merger").AddMetadata("clip.has_vision_encoder", true).AddMetadata("clip.use_gelu", true)
            .AddMetadata("clip.vision.block_count", (uint)P.LayerCount).AddMetadata("clip.vision.embedding_length", (uint)P.HiddenSize)
            .AddMetadata("clip.vision.feed_forward_length", (uint)P.FfnHiddenSize).AddMetadata("clip.vision.attention.head_count", (uint)P.HeadCount)
            .AddMetadata("clip.vision.attention.layer_norm_epsilon", 1e-6f).AddMetadata("clip.vision.patch_size", (uint)P.PatchSize)
            .AddMetadata("clip.vision.image_size", (uint)(P.PatchSize * P.PositionGridSide)).AddMetadata("clip.vision.spatial_merge_size", (uint)P.SpatialMergeSize)
            .AddMetadata("clip.vision.projection_dim", (uint)P.OutHiddenSize)
            .AddMetadata("clip.vision.image_mean", Mean).AddMetadata("clip.vision.image_std", Std)
            .AddMetadata("clip.vision.min_pixels", (uint)P.MinPixels).AddMetadata("clip.vision.max_pixels", (uint)P.MaxPixels)
            .AddMetadata("clip.vision.is_deepstack_layers", Enumerable.Repeat(false, P.LayerCount).ToArray());
        void Add(string name, long[] shape, float[] d) => w.AddTensor(name, DataType.F32, Bytes(d), shape);
        foreach (var (hfName, (shape, d)) in hf)
        {
            string n = hfName["model.visual.".Length..];
            string? g = n switch
            {
                "patch_embed.proj.bias" => "v.patch_embd.bias", "pos_embed.weight" => "v.position_embd.weight",
                "merger.norm.weight" => "v.post_ln.weight", "merger.norm.bias" => "v.post_ln.bias",
                "merger.linear_fc1.weight" => "mm.0.weight", "merger.linear_fc1.bias" => "mm.0.bias",
                "merger.linear_fc2.weight" => "mm.2.weight", "merger.linear_fc2.bias" => "mm.2.bias",
                _ => null,
            };
            if (n == "patch_embed.proj.weight")
            {
                // Conv3d [H, C, T, p, p] → one conv2d kernel [H, C, p, p] per temporal frame.
                long H = shape[0];
                long C = shape[1];
                long T = shape[2];
                long pp = shape[3] * shape[4];
                for (int t = 0; t < T; t++)
                {
                    var part = new float[H * C * pp];
                    for (long h = 0; h < H; h++)
                    {
                        for (long c = 0; c < C; c++)
                        {
                            Array.Copy(d, ((h * C + c) * T + t) * pp, part, (h * C + c) * pp, pp);
                        }
                    }

                    Add(t == 0 ? "v.patch_embd.weight" : $"v.patch_embd.weight.{t}", [H, C, shape[3], shape[4]], part);
                }
                continue;
            }
            if (g is null)
            {
                var parts = n.Split('.');   // blocks.{i}.xxx
                string layer = $"v.blk.{parts[1]}.";
                g = layer + string.Join('.', parts[2..]) switch
                {
                    "norm1.weight" => "ln1.weight", "norm1.bias" => "ln1.bias", "norm2.weight" => "ln2.weight", "norm2.bias" => "ln2.bias",
                    "attn.qkv.weight" => "attn_qkv.weight", "attn.qkv.bias" => "attn_qkv.bias", "attn.proj.weight" => "attn_out.weight", "attn.proj.bias" => "attn_out.bias",
                    "mlp.linear_fc1.weight" => "ffn_up.weight", "mlp.linear_fc1.bias" => "ffn_up.bias", "mlp.linear_fc2.weight" => "ffn_down.weight", "mlp.linear_fc2.bias" => "ffn_down.bias",
                    var x => throw new InvalidOperationException(x),
                };
            }
            Add(g, shape, d);
        }
        w.Write(path);
        return path;
    }

    private ModelSource Track(ModelSource s)
    {
        _open.AddRange(s.OwnedResources);
        return s;
    }

    [Fact]
    public void Hf_and_mmproj_importers_agree_and_preserve_the_weights()
    {
        var hf = HfTensors();
        var hfSrc = HfMapping(WriteHf(hf));
        var ggSrc = Track(GgufImporter.Open(WriteMmproj(hf)));

        Assert.Equal("qwen3_5", ggSrc.Architecture);
        var ph = VisionParameters.FromMetadata(hfSrc.Metadata, "qwen3_5.vision.")!;
        var pg = VisionParameters.FromMetadata(ggSrc.Metadata, "qwen3_5.vision.")!;
        Assert.Equal(P with { ImageMean = [], ImageStd = [] }, ph with { ImageMean = [], ImageStd = [] });
        Assert.Equal(ph with { ImageMean = [], ImageStd = [] }, pg with { ImageMean = [], ImageStd = [] });
        Assert.Equal(Mean, ph.ImageMean);
        Assert.Equal(Mean, pg.ImageMean);
        Assert.Equal(Std, ph.ImageStd);
        Assert.Equal(Std, pg.ImageStd);

        Assert.Equal(hfSrc.Keys.Order(), ggSrc.Weights.Keys.Order());
        Assert.Equal(Tiny.VisionWeights().Keys.Order(), ggSrc.Weights.Keys.Order());
        foreach (var (name, h) in hfSrc)
        {
            var g = ggSrc.Weights[name];
            Assert.Equal(h.Shape.ToString(), g.Shape.ToString());
            Assert.Equal(Tiny.VisionWeights()[name].Shape.ToString(), g.Shape.ToString());
            Assert.True(h.ReadAll().AsSpan().SequenceEqual(g.ReadAll()), $"{name} differs between importers");
        }
        // And the canonical patch weight really is the conv kernel flattened as (channel, frame, row, col): HF's raw layout.
        Assert.True(hfSrc["vision.patch_embed.weight"].ReadAll().AsSpan().SequenceEqual(Bytes(hf["model.visual.patch_embed.proj.weight"].Data)));
    }

    private HfView HfMapping(string dir)
    {
        using var config = HfConfig.Load(dir);
        var cp = SafetensorsCheckpoint.Open(dir);
        _open.Add(cp);
        var mapping = new Qwen35HfMapping();
        var md = new Dictionary<string, object>(mapping.MapHyperParameters(config, HfImportOptions.Default));
        return new HfView(md, mapping.MapWeights(config, cp.Tensors, HfImportOptions.Default).ToDictionary(k => k.Key, k => k.Value));
    }

    // Adapter so the test reads naturally: (metadata, weights) of the HF mapping.
    private sealed class HfView(IReadOnlyDictionary<string, object> md, Dictionary<string, WeightInfo> w) : IEnumerable<KeyValuePair<string, WeightInfo>>
    {
        public IReadOnlyDictionary<string, object> Metadata => md;
        public IEnumerable<string> Keys => w.Keys;
        public WeightInfo this[string k] => w[k];
        public IEnumerator<KeyValuePair<string, WeightInfo>> GetEnumerator() => w.GetEnumerator();
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }

    [Fact]
    public void Unsupported_towers_fail_loudly()
    {
        var hf = HfTensors();
        string gg = WriteMmproj(hf);
        // An unknown projector type has no mapping.
        var bytes = File.ReadAllBytes(gg);
        var idx = Encoding.ASCII.GetBytes("qwen3vl_merger");
        int at = bytes.AsSpan().IndexOf(idx);
        Assert.True(at > 0);
        Encoding.ASCII.GetBytes("mystery_merger").CopyTo(bytes, at);
        string bad = Path.Combine(_dir, "bad.gguf");
        File.WriteAllBytes(bad, bytes);
        Assert.Throws<NotSupportedException>(() => GgufImporter.Open(bad));
    }
}
