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

    /// <summary>
    /// Qwen2-VL-2B and Qwen2.5-VL-3B (Q4_K_M with F16 mmproj files from ggml-org) on the CPU and the GPU. llama-mtmd-cli b11177 (CPU, greedy)
    /// answers "The image contains two distinct geometric shapes: a red circle and a blue square. ..." (Qwen2-VL) and "The image consists of two
    /// primary shapes: a red circle and a blue rectangle. ..." (Qwen2.5-VL); the resized images differ slightly (llama.cpp's bicubic
    /// resize is not antialiased), so only the content is compared (Sharplight's Qwen2.5-VL calls the square a square).
    /// </summary>
    [Theory]
    [InlineData("qwen2-vl", false)]
    [InlineData("qwen2-vl", true)]
    [InlineData("qwen2.5-vl", false)]
    [InlineData("qwen2.5-vl", true)]
    public void Gguf_qwen2_vl_with_mmproj_describes_the_image(string which, bool gpu)
    {
        bool v25 = which == "qwen2.5-vl";
        string gguf = v25 ? Env.GgufQwen25Vl3b : Env.GgufQwen2Vl2b;
        string mmproj = v25 ? Env.MmprojQwen25Vl3b : Env.MmprojQwen2Vl2b;
        Assert.SkipUnless(File.Exists(gguf) && File.Exists(mmproj), $"{which} GGUF or mmproj not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = LlmModel.Load(gguf, Options(slot!, (AuxiliaryRoles.Mmproj, mmproj)));
        Assert.True(model.SupportsImages);
        var text = Describe(model, Env.Shapes());
        Console.Error.WriteLine("  -> " + text.ReplaceLineEndings(" "));
        Assert.Contains("red circle", text, StringComparison.OrdinalIgnoreCase);
        Assert.True(text.Contains("blue square", StringComparison.OrdinalIgnoreCase) || text.Contains("blue rectangle", StringComparison.OrdinalIgnoreCase), text);
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
        AssertMatchesTransformers(Env.HfVl2b, gpu, 4 * 2048, 114, Vl2bTop, Vl2bTopLogits, Vl2bGreedy);
    }

    // transformers 5 (Qwen2VLForConditionalGeneration, f32, tools/hf_vl_describe_reference.py) on the shapes image: 22×22 patches, 121 image rows.
    private static readonly int[] Qwen2Vl2bTop = [785, 1986, 641, 151648, 40];
    private static readonly float[] Qwen2Vl2bTopLogits = [23.4386f, 22.0056f, 19.3914f, 16.1989f, 16.1619f];
    private static readonly int[] Qwen2Vl2bGreedy = [785, 2168, 17167, 315, 1378, 52484, 20816, 25, 264, 2518, 12671, 323, 264, 6303, 9334, 13, 576, 2518, 12671, 374, 34228, 389, 279, 2115, 3108, 315, 279, 2168, 11, 1393, 279, 6303];

    /// <summary>Qwen2-VL-2B from its Hugging Face checkpoint (LayerNorm / quick-GELU tower, sectioned M-RoPE) against transformers.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Hf_qwen2_vl_2b_matches_transformers(bool gpu)
    {
        Assert.SkipUnless(Directory.Exists(Env.HfQwen2Vl2b), "Qwen2-VL-2B (HF) not available");
        AssertMatchesTransformers(Env.HfQwen2Vl2b, gpu, 1536, 146, Qwen2Vl2bTop, Qwen2Vl2bTopLogits, Qwen2Vl2bGreedy);
    }

    // transformers 5 (float32, CPU), Qwen2.5-VL-3B-Instruct: "The image consists of two simple geometric shapes placed side by side. On the
    // left, ...". Only the first 16 greedy ids are compared: at the 17th, transformers' cached generate picks "," while its own full forward
    // over the same prefix (like Sharplight) picks " side" (33.07 vs 32.86).
    private static readonly int[] Qwen25Vl3bTop = [785, 1986, 3862, 39814, 95456];
    private static readonly float[] Qwen25Vl3bTopLogits = [31.388f, 29.8497f, 24.4912f, 23.845f, 23.8003f];
    private static readonly int[] Qwen25Vl3bGreedy = [785, 2168, 17167, 315, 1378, 4285, 52484, 20816, 9099, 3108, 553, 3108, 13, 1913, 279, 2115];

    /// <summary>Qwen2.5-VL-3B from its Hugging Face checkpoint (RMS-norm / gated-SiLU tower with windowed attention) against transformers.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Hf_qwen2_5_vl_3b_matches_transformers(bool gpu)
    {
        Assert.SkipUnless(Directory.Exists(Env.HfQwen25Vl3b), "Qwen2.5-VL-3B (HF) not available");
        AssertMatchesTransformers(Env.HfQwen25Vl3b, gpu, 2048, 146, Qwen25Vl3bTop, Qwen25Vl3bTopLogits, Qwen25Vl3bGreedy);
    }

    /// <summary>The shapes image with "Describe this image." through the checkpoint's chat template: top-5 prompt logits and the greedy ids.</summary>
    private static void AssertMatchesTransformers(string dir, bool gpu, int embeddingSize, int rows, int[] expectedTop, float[] expectedLogits, int[] expectedGreedy)
    {
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = LlmModel.Load(dir, Options(slot!));
        using var vision = model.CreateVisionEncoder();
        Assert.Equal(embeddingSize, vision.EmbeddingSize);
        using var session = model.CreateSession(WellKnown.DecodeMultimodalGraph);
        var template = new ChatTemplate(model.ChatTemplate!, "", "");
        var prompt = template.Render([new ChatMessage("user", Parts: [ChatContentPart.Image(), ChatContentPart.FromText(Question)])], addGenerationPrompt: true);
        var tokens = model.Tokenizer.Encode(prompt, addSpecial: true, parseSpecial: true);
        var images = new[] { vision.Preprocess(Env.Shapes()) };
        var mm = MultimodalRegistry.Find(vision.Definition.Architecture)!.Assemble(new MultimodalContext(vision.Definition, t => Single(model, t)), tokens, images, 0);
        Assert.Equal(rows, mm.Length);
        var logits = session.Process(mm, images.Select(vision.Encode).ToList());
        var top = Enumerable.Range(0, logits.Length).OrderByDescending(i => logits[i]).Take(5).ToArray();
        Console.Error.WriteLine($"  top {string.Join(", ", top.Select(i => $"{i}:{logits[i]:F4}"))}");
        Assert.Equal(expectedTop, top);
        for (int i = 0; i < top.Length; i++)
        {
            Assert.True(Math.Abs(logits[top[i]] - expectedLogits[i]) < (gpu ? 0.15f : 0.02f), $"logit {top[i]}: {logits[top[i]]} vs {expectedLogits[i]}");
        }

        var ids = new List<int>();
        int next = top[0];
        while (ids.Count < expectedGreedy.Length)
        {
            ids.Add(next);
            var l = session.Process([next]);
            next = Enumerable.Range(0, l.Length).MaxBy(i => l[i]);
        }

        Assert.Equal(expectedGreedy, ids.ToArray());
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
