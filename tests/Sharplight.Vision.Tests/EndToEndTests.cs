using System.Text;
using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Templating;

namespace Sharplight.Vision.Tests;

/// <summary>Image → text with real models (skipped when the model files are absent): greedy descriptions of the shapes test image.</summary>
public class EndToEndTests
{
    private const string Question = "Describe this image.";

    private static string Describe(LlmModel model, ImageInput image, int maxTokens = 48)
    {
        using var vision = model.CreateVisionEncoder();
        using var session = model.CreateSession(WellKnown.DecodeMultimodalGraph);
        var gen = new Generator(session, model.Tokenizer, vision);
        var template = new ChatTemplate(model.ChatTemplate!, "", "");
        var prompt = template.Render([new ChatMessage("user", Parts: [ChatContentPart.Image(), ChatContentPart.FromText(Question)])], addGenerationPrompt: true,
            new Dictionary<string, object?> { ["enable_thinking"] = false });
        var sb = new StringBuilder();
        foreach (var t in gen.Generate(prompt, [image], new GenerationOptions { MaxNewTokens = maxTokens, Sampling = SamplingOptions.Greedy }))
        {
            sb.Append(t.Text);
        }

        return sb.ToString();
    }

    private static void AssertShapes(string text)
    {
        Console.Error.WriteLine("  -> " + text.ReplaceLineEndings(" ")[..Math.Min(160, text.Length)]);
        Assert.Contains("red", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("blue", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("circle", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("square", text, StringComparison.OrdinalIgnoreCase);
    }

    private static LlmLoadOptions Options(DeviceSlot slot, params (string, string)[] aux) => new() { Devices = [slot], ContextLength = 1024, MaxBatchTokens = 256, Auxiliary = aux, ImageTokens = new ImageTokenLimits(null, 256) };

    [Fact]
    public void Hf_0_8b_on_cpu_describes_the_image_and_survives_a_package_round_trip()
    {
        Assert.SkipUnless(Directory.Exists(Env.Hf08), "Qwen3.5-0.8B (HF) not available");
        string text;
        using (var model = LlmModel.Load(Env.Hf08, Options(Env.Cpu())))
        {
            text = Describe(model, Env.Shapes());
            AssertShapes(text);
            var tmp = Path.Combine(Path.GetTempPath(), "slvision-e2e-" + Guid.NewGuid().ToString("N")[..8] + ModelPackage.Extension);
            try
            {
                ModelPackage.Save(tmp, model.Definition, []);
                using var loaded = LlmModel.Load(tmp, Options(Env.Cpu()));
                Assert.True(loaded.SupportsImages);
                Assert.Equal(text, Describe(loaded, Env.Shapes()));   // identical weights, graphs and preprocessing: identical greedy text
            }
            finally { File.Delete(tmp); }
        }
    }

    [Fact]
    public void Gguf_0_8b_with_mmproj_on_cpu_describes_the_image()
    {
        Assert.SkipUnless(File.Exists(Env.Gguf08) && File.Exists(Env.Mmproj08), "Qwen3.5-0.8B GGUF or its converted mmproj (.work/mmproj-0.8B-f32.gguf) not available");
        using var model = LlmModel.Load(Env.Gguf08, Options(Env.Cpu(), (AuxiliaryRoles.Mmproj, Env.Mmproj08)));
        Assert.True(model.SupportsImages);
        AssertShapes(Describe(model, Env.Shapes()));
    }

    /// <summary>Qwen3-VL (generic decoder with interleaved M-RoPE) with a DeepStack tower from its mmproj, on the CPU and the GPU.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Gguf_qwen3_vl_2b_with_deepstack_mmproj_describes_the_image(bool gpu)
    {
        Assert.SkipUnless(File.Exists(Env.GgufVl2b) && File.Exists(Env.MmprojVl2b), "Qwen3-VL-2B GGUF or mmproj not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = LlmModel.Load(Env.GgufVl2b, Options(slot!, (AuxiliaryRoles.Mmproj, Env.MmprojVl2b)));
        Assert.True(model.SupportsImages);
        AssertShapes(Describe(model, Env.Shapes()));
    }

    // transformers 5 (Qwen3VLForConditionalGeneration, f32) on the shapes image with the same prompt: the last prompt row's top-5 logits
    // and the greedy continuation.
    private static readonly int[] Vl2bTop = [1986, 785, 28715, 2124, 8420];
    private static readonly float[] Vl2bTopLogits = [37.6835f, 35.2783f, 30.8774f, 29.6936f, 28.3085f];
    private static readonly int[] Vl2bGreedy = [1986, 2168, 18689, 1378, 52484, 20816, 389, 264, 14396, 4158, 4004, 13, 576, 20816, 525, 264, 2518, 12671, 323, 264, 6303, 9334, 382, 12, 576, 2518, 12671, 374, 34228, 389, 279, 2115];

    /// <summary>Qwen3-VL-2B from its Hugging Face checkpoint (tower with DeepStack mergers in the same source) against transformers.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Hf_qwen3_vl_2b_matches_transformers(bool gpu)
    {
        Assert.SkipUnless(Directory.Exists(Env.HfVl2b), "Qwen3-VL-2B (HF) not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = LlmModel.Load(Env.HfVl2b, Options(slot!));
        using var vision = model.CreateVisionEncoder();
        Assert.Equal(4 * 2048, vision.EmbeddingSize);
        using var session = model.CreateSession(WellKnown.DecodeMultimodalGraph);
        var template = new ChatTemplate(model.ChatTemplate!, "", "");
        var prompt = template.Render([new ChatMessage("user", Parts: [ChatContentPart.Image(), ChatContentPart.FromText(Question)])], addGenerationPrompt: true);
        var tokens = model.Tokenizer.Encode(prompt, addSpecial: true, parseSpecial: true);
        var images = new[] { vision.Preprocess(Env.Shapes()) };
        var mm = MultimodalRegistry.Find(vision.Definition.Architecture)!.Assemble(new MultimodalContext(vision.Definition, t => Single(model, t)), tokens, images, 0);
        Assert.Equal(114, mm.Length);
        var logits = session.Process(mm, images.Select(vision.Encode).ToList());
        var top = Enumerable.Range(0, logits.Length).OrderByDescending(i => logits[i]).Take(5).ToArray();
        Console.Error.WriteLine($"  top {string.Join(", ", top.Select(i => $"{i}:{logits[i]:F4}"))}");
        Assert.Equal(Vl2bTop, top);
        for (int i = 0; i < top.Length; i++)
        {
            Assert.True(Math.Abs(logits[top[i]] - Vl2bTopLogits[i]) < (gpu ? 0.15f : 0.02f), $"logit {top[i]}: {logits[top[i]]} vs {Vl2bTopLogits[i]}");
        }

        var ids = new List<int>();
        int next = top[0];
        while (ids.Count < Vl2bGreedy.Length)
        {
            ids.Add(next);
            var l = session.Process([next]);
            next = Enumerable.Range(0, l.Length).MaxBy(i => l[i]);
        }

        Assert.Equal(Vl2bGreedy, ids.ToArray());
    }

    private static int? Single(LlmModel model, string text)
    {
        var ids = model.Tokenizer.Encode(text, addSpecial: false, parseSpecial: true);
        return ids.Length == 1 ? ids[0] : null;
    }

    [Fact]
    public void Gguf_without_mmproj_has_no_images()
    {
        Assert.SkipUnless(File.Exists(Env.Gguf08), "Qwen3.5-0.8B GGUF not available");
        using var model = LlmModel.Load(Env.Gguf08, Options(Env.Cpu()));
        Assert.False(model.SupportsImages);
        Assert.Throws<NotSupportedException>(() => model.CreateVisionEncoder());
    }

    [Fact]
    public void Hf_0_8b_on_vulkan_describes_the_image()
    {
        Assert.SkipUnless(Directory.Exists(Env.Hf08), "Qwen3.5-0.8B (HF) not available");
        var slot = Env.Vulkan();
        Assert.SkipUnless(slot is not null, "no Vulkan device");
        using var model = LlmModel.Load(Env.Hf08, Options(slot!));
        AssertShapes(Describe(model, Env.Shapes()));
    }

    /// <summary>Opt-in (SHARPLIGHT_TEST_27B=1): the 27B GGUF with its mmproj on the Vulkan device.</summary>
    [Fact]
    public void Gguf_27b_with_mmproj_on_vulkan_describes_the_image()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_27B") == "1", "set SHARPLIGHT_TEST_27B=1 to run the 27B end-to-end test");
        Assert.SkipUnless(File.Exists(Env.Gguf27) && File.Exists(Env.Mmproj27), "Qwen3.8-27B GGUF or mmproj not available");
        var slot = Env.Vulkan();
        Assert.SkipUnless(slot is not null, "no Vulkan device");
        using var model = LlmModel.Load(Env.Gguf27, Options(slot!, (AuxiliaryRoles.Mmproj, Env.Mmproj27)));
        AssertShapes(Describe(model, Env.Shapes()));
    }

    /// <summary>Text-only generation on the multimodal graph is unchanged (zero mask, no image).</summary>
    [Fact]
    public void Text_only_prompts_work_on_the_multimodal_session()
    {
        Assert.SkipUnless(Directory.Exists(Env.Hf08), "Qwen3.5-0.8B (HF) not available");
        using var model = LlmModel.Load(Env.Hf08, Options(Env.Cpu()));
        string Run(string graph)
        {
            using var session = model.CreateSession(graph);
            var sb = new StringBuilder();
            foreach (var t in new Generator(session, model.Tokenizer).Generate("The capital of France is", new GenerationOptions { MaxNewTokens = 8, Sampling = SamplingOptions.Greedy }))
            {
                sb.Append(t.Text);
            }

            return sb.ToString();
        }
        var plain = Run(WellKnown.DecodeGraph);
        Assert.Contains("Paris", plain);
        Assert.Equal(plain, Run(WellKnown.DecodeMultimodalGraph));
    }

    /// <summary>Bidirectional attention on GPU vs CPU, including the 27B tower's head size 72.</summary>
    [Theory]
    [InlineData(37, 4, 64)]
    [InlineData(100, 16, 72)]
    public void Bidirectional_attention_gpu_matches_cpu(int T, int heads, int D)
    {
        var gpu = Env.Vulkan();
        Assert.SkipUnless(gpu is not null, "no Vulkan device");
        var b = new GraphBuilder("attn", _ => null);
        var Tn = Dim.Sym("T");
        var q = b.Input("q", InputRole.Other, DataType.F32, new Shape(Tn, heads, D));
        var k = b.Input("k", InputRole.Other, DataType.F32, new Shape(Tn, heads, D));
        var v = b.Input("v", InputRole.Other, DataType.F32, new Shape(Tn, heads, D));
        b.Output("o", b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(D)), ("causal", false))));
        var g = b.Build();
        var rng = new Random(4);
        float[] Rand(int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] qd = Rand(T * heads * D);
        float[] kd = Rand(T * heads * D);
        float[] vd = Rand(T * heads * D);
        float[] Run(DeviceSlot slot)
        {
            using var cm = CompiledModel.Compile(g, new PlacementOptions { Devices = [slot], Compile = new CompileOptions { MaxTokens = 128 } });
            cm.SetInput<float>(g.Inputs[0], qd);
            cm.SetInput<float>(g.Inputs[1], kd);
            cm.SetInput<float>(g.Inputs[2], vd);
            var run = new RunParameters();
            run.Symbols["T"] = T;
            cm.Run(run);
            var o = new float[T * heads * D];
            cm.GetOutput<float>("o", o);
            return o;
        }
        var (abs, _, rms) = Env.Compare(Run(gpu!), Run(Env.Cpu()));
        // GPUs with f16 matrix units run attention as tensor-core flash attention (f16 Q/K/V and probabilities).
        Assert.True(rms < 1e-3, $"rel RMS {rms}, max abs {abs}");
    }
}
