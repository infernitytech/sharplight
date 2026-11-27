using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;
using Sharplight.Templating;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Image prompts through the batching engine (<see cref="InferenceEngine"/> over a decoder created with media input, <see cref="ImagePrompts"/>):
/// several image and text requests decoded together give exactly the greedy tokens of the single-sequence path (a <see cref="Generator"/> over a
/// session of the multimodal graph with the vision encoder), and a repeated image prompt is neither encoded nor prefilled again (its blocks come
/// from the prefix cache). Skipped when the model files are not available.
/// </summary>
public class BatchedMultimodalTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private const int Tokens = 24;

    public static TheoryData<string, bool> Cases => new()
    {
        { "qwen2-vl-tiny", false }, { "qwen2-vl-tiny", true },
        { "qwen2.5-vl-tiny", false }, { "qwen2.5-vl-tiny", true },
        { "qwen3-vl-moe-tiny", false }, { "qwen3-vl-moe-tiny", true },
        { "glm4v-tiny", false }, { "glm4v-tiny", true },
        { "qwen3-vl-2b", false }, { "qwen3-vl-2b", true },
        { "qwen3.5-0.8b", false }, { "qwen3.5-0.8b", true },
        { "glm-ocr", false }, { "glm-ocr", true },
        { "gemma3-vl-tiny", false }, { "gemma3-vl-tiny", true },
    };

    /// <summary>The checkpoint (and mmproj) of a case, and the most image tokens it is loaded with.</summary>
    internal static (string Path, string? Mmproj, int ImageTokens) Model(string name) => name switch
    {
        "qwen2-vl-tiny" => (Path.Combine(Home, "models/hf/tiny-random-qwen2-vl-wide"), null, 128),
        "qwen2.5-vl-tiny" => (Path.Combine(Home, "models/hf/tiny-random-qwen2-5-vl-wide"), null, 128),
        "qwen3-vl-moe-tiny" => (Path.Combine(Home, "models/hf/tiny-random-qwen3-vl-moe-wide"), null, 128),
        "glm4v-tiny" => (Path.Combine(Home, "models/hf/tiny-random-glm4v"), null, 128),
        "qwen3-vl-2b" => (Path.Combine(Home, "models/gguf/Qwen3-VL-2B-Instruct/Qwen3VL-2B-Instruct-Q4_K_M.gguf"),
            Path.Combine(Home, "models/gguf/Qwen3-VL-2B-Instruct/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"), 256),
        "qwen3.5-0.8b" => (Env.Hf08, null, 256),
        "glm-ocr" => (Path.Combine(Home, "models/vision-r10/glm-ocr"), null, 256),
        // Bidirectional image rows (issue #98): the engine keeps an image's rows in one step.
        "gemma3-vl-tiny" => (Path.Combine(Home, "models/gguf/tiny-random-gemma3-vl-f32.gguf"), Path.Combine(Home, "models/gguf/mmproj-tiny-random-gemma3-vl-f32.gguf"), 128),
        _ => throw new ArgumentException(name),
    };

    /// <summary>Loads a case's model, or skips the test when its files are missing.</summary>
    internal static LlmModel Load(string name, bool gpu)
    {
        var (path, mmproj, imageTokens) = Model(name);
        bool present = File.Exists(path) || (Directory.Exists(path) && Directory.EnumerateFiles(path, "*.safetensors").Any());
        Assert.SkipUnless(present, $"{name} not available");
        Assert.SkipWhen(mmproj is not null && !File.Exists(mmproj), $"{name}: mmproj not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        var options = new LlmLoadOptions { Devices = [slot!], ContextLength = 1024, MaxBatchTokens = 256, ImageTokens = new ImageTokenLimits(null, imageTokens) };
        if (mmproj is not null)
        {
            options.Auxiliary[AuxiliaryRoles.Mmproj] = mmproj;
        }

        return LlmModel.Load(TestHost.Loader, path, options);
    }

    /// <summary>A user turn with an image placeholder and a question, as the model's chat template renders it.</summary>
    internal static string ImagePrompt(LlmModel model, string question)
    {
        string token = VisionParameters.Of(model.Definition)?.ImageToken ?? VisionParameters.DefaultImageToken;
        if (model.ChatTemplate is { } source)
        {
            try
            {
                var template = new ChatTemplate(source, "", "");
                string rendered = template.Render([new ChatMessage("user", Parts: [ChatContentPart.Image(), ChatContentPart.FromText(question)])], addGenerationPrompt: true,
                    new Dictionary<string, object?> { ["enable_thinking"] = false });
                if (rendered.Contains(token, StringComparison.Ordinal))
                {
                    return rendered;
                }
            }
            catch (TemplateRuntimeException)
            {
                // A template that takes string content only.
            }
        }

        // Tiny checkpoints whose template does not render image parts: the placeholder before the question.
        return $"{token}{question}";
    }

    /// <summary>The shapes test image mirrored left to right (a second, different image).</summary>
    internal static ImageInput Mirrored(ImageInput image)
    {
        var rgb = new byte[image.Rgb.Length];
        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                int from = (y * image.Width + x) * 3;
                int to = (y * image.Width + image.Width - 1 - x) * 3;
                rgb[to] = image.Rgb[from];
                rgb[to + 1] = image.Rgb[from + 1];
                rgb[to + 2] = image.Rgb[from + 2];
            }
        }

        return new ImageInput(image.Width, image.Height, rgb);
    }

    /// <summary>The single-sequence path: a generator over a fresh session (multimodal with the vision encoder for image prompts).</summary>
    internal static int[] SessionGreedy(LlmModel model, VisionEncoder vision, string prompt, ImageInput? image, int count, bool stopAtEog = false)
    {
        using var session = model.CreateSession(image is null ? WellKnown.DecodeGraph : WellKnown.DecodeMultimodalGraph);
        var generator = new Generator(session, model.Tokenizer, vision);
        var options = new GenerationOptions { MaxNewTokens = count, Sampling = SamplingOptions.Greedy, StopAtEog = stopAtEog };
        var tokens = image is null ? generator.Generate(prompt, options) : generator.Generate(prompt, [image], options);
        return [.. tokens.Where(t => t.Id >= 0).Select(t => t.Id)];
    }

    internal static GenerationRequest Request(LlmModel model, ImagePrompts images, string prompt, ImageInput? image, int count, bool stopAtEog = false)
    {
        var request = new GenerationRequest
        {
            PromptTokens = model.Tokenizer.Encode(prompt, addSpecial: true, parseSpecial: true),
            MaxTokens = count, StopAtEog = stopAtEog, Logprobs = true, Sampling = SamplingParams.GreedyDefault,
        };
        return image is null ? request : images.WithImages(request, [image]);
    }

    internal static async Task<(int[] Ids, GenerationUsage Usage)[]> Run(InferenceEngine engine, IEnumerable<GenerationRequest> requests, CancellationToken ct)
    {
        var handles = requests.Select(r => engine.Submit(r, ct)).ToList();
        var results = await Task.WhenAll(handles.Select(h => h.CompleteAsync(ct)));
        foreach (var r in results)
        {
            Assert.Null(r.Error);
        }

        return [.. results.Select(r => (r.Logprobs.Select(l => l.TokenId).ToArray(), r.Usage))];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task Image_and_text_requests_batched_match_the_single_session_path(string name, bool gpu)
    {
        var ct = TestContext.Current.CancellationToken;
        using var model = Load(name, gpu);
        Assert.True(model.SupportsMediaInput);
        using var vision = model.CreateVisionEncoder();
        var shapes = Env.Shapes();
        var mirrored = Mirrored(shapes);
        var cases = new (string Prompt, ImageInput? Image)[]
        {
            (ImagePrompt(model, "Describe this image."), shapes),
            (ImagePrompt(model, "What colours do you see?"), mirrored),
            (model.Tokenizer.Decode(model.Tokenizer.Encode("The capital of France is", addSpecial: false, parseSpecial: false)), null),
            (ImagePrompt(model, "Describe this image."), shapes),
        };
        var expected = cases.Select(c => SessionGreedy(model, vision, c.Prompt, c.Image, Tokens)).ToArray();

        using var images = new ImagePrompts(vision, model.Tokenizer);
        using var engine = new InferenceEngine(model.CreateBatchDecoder(slots: 4, mediaInput: true), model.Tokenizer);
        var requests = cases.Select(c => Request(model, images, c.Prompt, c.Image, Tokens)).ToList();
        var actual = await Run(engine, requests, ct);
        for (int i = 0; i < cases.Length; i++)
        {
            TestContext.Current.SendDiagnosticMessage($"{name} ({(gpu ? "Vulkan" : "CPU")}) request {i}: {string.Join(' ', actual[i].Ids)}");
            AssertGreedyEqual(model, vision, cases[i].Prompt, cases[i].Image, expected[i], actual[i].Ids);
        }

        // The first image prompt again: at least every full block of it comes from the prefix cache (the slot that held it may give back
        // all but its last token), image rows included; nothing is encoded.
        long encoded = engine.GetMetrics().MediaEncoded;
        Assert.InRange(encoded, 1, 2);
        var again = await Run(engine, [requests[0]], ct);
        AssertGreedyEqual(model, vision, cases[0].Prompt, cases[0].Image, expected[0], again[0].Ids);
        int blocks = (requests[0].PromptTokens.Length - 1) / MultiSequenceDecoder.DefaultKvBlockSize * MultiSequenceDecoder.DefaultKvBlockSize;
        Assert.InRange(again[0].Usage.CachedTokens, blocks, requests[0].PromptTokens.Length - 1);
        Assert.Equal(encoded, engine.GetMetrics().MediaEncoded);
    }

    /// <summary>Largest top-1 / top-2 logit margin below which the batched kernels' accumulation order may pick the other token on a GPU.</summary>
    private const float NearTie = 0.1f;

    /// <summary>
    /// The engine's tokens equal the single-session path's, except that at a near tie of the single-session logits (a top-2 margin below
    /// <see cref="NearTie"/>, where the GPU's batched kernels may round the other way) the rest is not compared.
    /// </summary>
    private static void AssertGreedyEqual(LlmModel model, VisionEncoder vision, string prompt, ImageInput? image, int[] expected, int[] actual)
    {
        int at = 0;
        while (at < expected.Length && at < actual.Length && expected[at] == actual[at])
        {
            at++;
        }

        if (at == expected.Length && at == actual.Length)
        {
            return;
        }

        float margin = SessionMargin(model, vision, prompt, image, at);
        TestContext.Current.SendDiagnosticMessage($"first difference at token {at}: {expected.ElementAtOrDefault(at)} vs {actual.ElementAtOrDefault(at)}, single-session top-2 margin {margin:F4}");
        Assert.True(margin < NearTie, $"the tokens differ at {at} (top-2 margin {margin:F4}): expected {string.Join(' ', expected)}, actual {string.Join(' ', actual)}");
    }

    /// <summary>The single-session path's top-1 minus top-2 logit at generated token <paramref name="index"/>.</summary>
    private static float SessionMargin(LlmModel model, VisionEncoder vision, string prompt, ImageInput? image, int index)
    {
        using var session = model.CreateSession(image is null ? WellKnown.DecodeGraph : WellKnown.DecodeMultimodalGraph);
        var tokens = model.Tokenizer.Encode(prompt, addSpecial: true, parseSpecial: true);
        float[] logits;
        if (image is null)
        {
            logits = session.Process(tokens);
        }
        else
        {
            var pre = vision.Preprocess(image);
            var mm = vision.Multimodal!.Assemble(new MultimodalContext(model.Definition, t => model.Tokenizer.Encode(t, addSpecial: false, parseSpecial: true) is [var id] ? id : null),
                tokens, [pre], 0);
            logits = session.Process(mm, [vision.Encode(pre)]);
        }

        for (int i = 0; ; i++)
        {
            var order = Enumerable.Range(0, logits.Length).OrderByDescending(k => logits[k]).Take(2).ToArray();
            if (i == index)
            {
                return logits[order[0]] - logits[order[1]];
            }

            logits = session.Process([order[0]]);
        }
    }

    /// <summary>
    /// GLM-OCR's multi-token-prediction layer (<c>layers.16</c>) as the engine's draft head on an image prompt: the head runs the target's
    /// rows with their M-RoPE positions and the image rows' embeddings (<see cref="WellKnown.DraftMultimodalGraph"/>), the output is token for
    /// token the single-session path's, and most drafts are accepted, which shortens decoding.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Glm_ocr_mtp_head_speculates_on_image_prompts_with_identical_output(bool gpu)
    {
        var ct = TestContext.Current.CancellationToken;
        string sample = Path.Combine(Model("glm-ocr").Path, "sharplight-sample.png");
        Assert.SkipUnless(File.Exists(sample) && File.Exists(Path.Combine(Model("glm-ocr").Path, "model.safetensors")), "GLM-OCR or its sample image is not available");
        using var model = Load("glm-ocr", gpu);
        Assert.True(model.HasDraftHead, "GLM-OCR's MTP layer is imported as the draft head");
        Assert.True(model.Definition.Graphs.ContainsKey(WellKnown.DraftMultimodalGraph));
        using var vision = model.CreateVisionEncoder();
        var image = Sharplight.Imaging.PngCodec.Read(sample);
        string prompt = ImagePrompt(model, "Text Recognition:");
        const int Count = 128;
        const int Drafts = 3;
        const int Slots = 2;
        var expected = SessionGreedy(model, vision, prompt, image, Count, stopAtEog: true);
        using var images = new ImagePrompts(vision, model.Tokenizer);
        var request = Request(model, images, prompt, image, Count, stopAtEog: true);

        GenerationTimings plainTimings, mtpTimings;
        using (var plain = new InferenceEngine(model.CreateBatchDecoder(Slots, mediaInput: true), model.Tokenizer))
        {
            await Run(plain, [request], ct);   // warm-up: kernels and command buffers of every step shape
            (var ids, plainTimings) = await RunTimed(plain, request, ct);
            Assert.Equal(expected, ids);
        }

        var speculator = model.CreateMtpSpeculator(Slots, Drafts, mediaInput: true);
        var decoder = model.CreateBatchDecoder(Slots, maxLogitsRows: Slots * (Drafts + 1), draftTokens: Drafts, mediaInput: true);
        using var engine = new InferenceEngine(decoder, model.Tokenizer, new EngineOptions { Speculator = speculator });
        Assert.True(engine.Speculates);
        await Run(engine, [request], ct);
        (var specIds, mtpTimings) = await RunTimed(engine, request, ct);
        Assert.Equal(expected, specIds);
        double acceptance = mtpTimings.DraftTokens == 0 ? 0 : (double)mtpTimings.AcceptedDraftTokens / mtpTimings.DraftTokens;
        double plainRate = (expected.Length - 1) / plainTimings.Decode.TotalSeconds;
        double mtpRate = (expected.Length - 1) / mtpTimings.Decode.TotalSeconds;
        TestContext.Current.SendDiagnosticMessage($"GLM-OCR MTP ({(gpu ? "Vulkan" : "CPU")}): {expected.Length} tokens, acceptance {acceptance:P0} "
            + $"({mtpTimings.AcceptedDraftTokens}/{mtpTimings.DraftTokens}), decode {plainRate:F1} -> {mtpRate:F1} tok/s ({mtpRate / plainRate:F2}x)");
        Assert.True(acceptance > 0.5, $"draft acceptance {acceptance:P0}");
    }

    private static async Task<(int[] Ids, GenerationTimings Timings)> RunTimed(InferenceEngine engine, GenerationRequest request, CancellationToken ct)
    {
        var ids = new List<int>();
        GenerationTimings timings = default;
        await foreach (var chunk in engine.Submit(request, ct).ReadAllAsync(ct))
        {
            Assert.Null(chunk.Error);
            ids.AddRange(chunk.Logprobs?.Select(l => l.TokenId) ?? []);
            if (chunk.Timings is { } t)
            {
                timings = t;
            }
        }

        return ([.. ids], timings);
    }
}
