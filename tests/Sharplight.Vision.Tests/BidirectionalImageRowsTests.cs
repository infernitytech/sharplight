using System.Text.Json;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Models;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Gemma 3's image rows attend to each other (mmproj <c>image_attention</c> = bidirectional, the multimodal graph's
/// <see cref="InputRole.AttentionLookahead"/> input; issue #98): the batch decoder (multi-sequence graph) gives the logits of transformers'
/// <c>token_type_ids</c> mask (reference_image.json of siglip_vl_reference.py) when the image prompt is prefilled in one step next to a text
/// sequence, and when a text-only first chunk (the plain kernels) precedes the chunk with the image. The single-session path is covered by
/// <see cref="TinyQwenVlImageTests"/>.
/// </summary>
public class BidirectionalImageRowsTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private const string Name = "tiny-random-gemma3-vl";

    [Theory]
    [InlineData(false, 0)]
    [InlineData(true, 0)]
    [InlineData(false, 3)]
    [InlineData(true, 3)]
    public void Batch_decoder_image_rows_match_the_token_type_mask(bool gpu, int firstChunk)
    {
        string refPath = Path.Combine(Home, "models/hf", Name, "reference_image.json");
        string gguf = Path.Combine(Home, "models/gguf", $"{Name}-f32.gguf");
        string mmproj = Path.Combine(Home, "models/gguf", $"mmproj-{Name}-f32.gguf");
        Assert.SkipUnless(File.Exists(refPath) && File.Exists(gguf) && File.Exists(mmproj), $"{Name}: reference_image.json, GGUF or mmproj not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = LlmModel.Load(TestHost.Loader, gguf, new LlmLoadOptions
        {
            Devices = [slot!], ContextLength = 256, Auxiliary = { [AuxiliaryRoles.Mmproj] = mmproj }, ImageTokens = new ImageTokenLimits(null, 128),
        });
        Assert.Equal(VisionParameters.BidirectionalImageAttention, VisionParameters.Of(model.Definition)!.ImageAttention);

        using var doc = JsonDocument.Parse(File.ReadAllBytes(refPath));
        var r = doc.RootElement;
        int side = r.GetProperty("width").GetInt32();
        using var vision = model.CreateVisionEncoder();
        var image = vision.Preprocessor.Patchify(TinyQwenVlImageTests.SineImage(side, side), side, side);
        var embeddings = vision.Encode(image);
        var promptIds = r.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        Assert.True(model.Architectures.TryFindMultimodal(model.Definition.Architecture, out var multimodal));
        var mm = multimodal.Assemble(new MultimodalContext(model.Definition, t => Single(model, t)), promptIds, [image], 0);
        int imageStart = Array.IndexOf(mm.ImageIndex, 0);
        int imageRows = mm.ImageIndex.Count(i => i == 0);
        Assert.True(imageRows > 1 && firstChunk <= imageStart);

        using var decoder = model.CreateBatchDecoder(slots: 2, mediaInput: true);
        Assert.True(decoder.BidirectionalMediaRows);
        int width = decoder.EmbeddingWidth;
        var other = model.Tokenizer.Encode("The capital of France is", addSpecial: true, parseSpecial: false);
        if (firstChunk > 0)
        {
            // A text-only first chunk of the image prompt next to a text sequence: no row looks ahead.
            decoder.Step([new SequenceInput(0, mm.Tokens.AsMemory(0, firstChunk), WantLogits: false), new SequenceInput(1, other, WantLogits: false)]);
        }

        // The rest of the image prompt (all image rows in this input) and a decode-like row of the text sequence in the same step.
        var rest = new SequenceInput(0, mm.Tokens.AsMemory(firstChunk))
        {
            Embeddings = [new EmbeddingRun(imageStart - firstChunk, embeddings.AsMemory(0, imageRows * width))],
        };
        var text = firstChunk > 0 ? new SequenceInput(1, new[] { other[^1] }) : new SequenceInput(1, other);
        var logits = decoder.Step([rest, text])[0].ToArray();

        var topIds = r.GetProperty("top_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var topLogits = r.GetProperty("top_logits").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        for (int i = 0; i < 5; i++)
        {
            Assert.True(Math.Abs(logits[topIds[i]] - topLogits[i]) < (gpu ? 2e-2f : 1e-2f), $"logit {topIds[i]}: {logits[topIds[i]]} vs {topLogits[i]}");
        }

        // The next decode step (no row looks ahead) continues the transformers greedy sequence.
        var greedy = r.GetProperty("greedy").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int next = Enumerable.Range(0, logits.Length).MaxBy(i => logits[i]);
        Assert.Equal(greedy[0], next);
        var step = decoder.Step([new SequenceInput(0, new[] { next })])[0].ToArray();
        Assert.Equal(greedy[1], Enumerable.Range(0, step.Length).MaxBy(i => step[i]));
    }

    private static int? Single(LlmModel model, string text)
    {
        var ids = model.Tokenizer.Encode(text, addSpecial: false, parseSpecial: true);
        return ids.Length == 1 ? ids[0] : null;
    }
}
