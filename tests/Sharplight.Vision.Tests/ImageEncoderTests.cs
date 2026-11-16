using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision.Tests;

/// <summary>
/// ViT / DINOv2 image encoders through the task layer, against transformers (<c>tools/vision-reference.py</c> writes the reference next to
/// the checkpoint). Skipped when the checkpoint or its reference is not present.
/// </summary>
public class ImageEncoderTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    private static string Model(string name) => Path.Combine(Home, Directory.Exists(Path.Combine(Home, "models/vision", name)) ? "models/vision" : "models/vision-sweep", name);

    private static bool Available(string name) => File.Exists(Path.Combine(Model(name), "sharplight-reference.json"));

    private static JsonElement Reference(string name) => JsonDocument.Parse(File.ReadAllText(Path.Combine(Model(name), "sharplight-reference.json"))).RootElement;

    private static float[] Floats(JsonElement e) => [.. e.EnumerateArray().Select(v => v.GetSingle())];

    private static TaskModel Load(string name, DeviceSlot device)
        => TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model(name), Devices = [device] });

    private static IEnumerable<(string Backend, DeviceSlot Slot)> Backends()
    {
        yield return ("cpu", Env.Cpu());
        if (Env.Vulkan() is { } vulkan)
        {
            yield return ("vulkan", vulkan);
        }
    }

    [Theory]
    [InlineData("vit-base-patch16-224")]
    [InlineData("nsfw_image_detection")]
    [InlineData("resnet50.a1_in1k")]
    [InlineData("resnet18.a1_in1k")]
    [InlineData("resnet34.a1_in1k")]
    [InlineData("wide_resnet50_2.racm_in1k")]
    public async Task VitClassifierMatchesTransformers(string name)
    {
        if (!Available(name))
        {
            return;
        }

        var reference = Reference(name);
        var expected = Floats(reference.GetProperty("logits"));
        var image = PngCodec.Read(Path.Combine(Model(name), "sharplight-sample.png"));
        foreach (var (backend, slot) in Backends())
        {
            using var model = Load(name, slot);
            Assert.Equal("image-classification", model.Description.Kind);
            using var classifier = new TaskClientProvider([.. Env.TaskFactories()]).Create<IImageClassifier>(model);
            var result = await classifier.ClassifyAsync(image, new ImageClassificationOptions { TopK = 0 });
            var logits = new float[expected.Length];
            foreach (var p in result.Predictions)
            {
                logits[p.Index] = p.Logit;
            }

            var (maxAbs, _, rms) = Env.Compare(logits, expected);
            Assert.True(rms < 2e-3, $"{backend}: logits relative rms {rms}, max abs {maxAbs}");
            Assert.Equal(Array.IndexOf(expected, expected.Max()), result.Top!.Index);
        }
    }

    [Theory]
    [InlineData("dinov2-small")]
    [InlineData("dinov2-base")]
    [InlineData("dinov2-with-registers-base")]
    [InlineData("dino-vitb16")]
    [InlineData("vit-base-patch16-224-in21k")]
    public async Task Dinov2EmbeddingMatchesTransformers(string name)
    {
        if (!Available(name))
        {
            return;
        }

        var reference = Reference(name);
        var expected = Floats(reference.GetProperty("pooled"));
        var image = PngCodec.Read(Path.Combine(Model(name), "sharplight-sample.png"));
        foreach (var (backend, slot) in Backends())
        {
            using var model = Load(name, slot);
            Assert.Equal("image-embedding", model.Description.Kind);
            using var embedder = new TaskClientProvider([.. Env.TaskFactories()]).Create<IImageEmbedder>(model);
            var embedding = await embedder.EmbedImageAsync(image, new ImageEmbeddingOptions { Normalize = false });
            Assert.Equal(expected.Length, embedder.Dimensions);
            var (maxAbs, _, rms) = Env.Compare(embedding, expected);
            Assert.True(rms < 2e-3, $"{backend}: embedding relative rms {rms}, max abs {maxAbs}");
        }
    }

    [Theory]
    [InlineData("clip-vit-base-patch32")]
    [InlineData("siglip-base-patch16-224")]
    [InlineData("clip-vit-base-patch16")]
    [InlineData("clip-vit-large-patch14")]
    [InlineData("siglip2-base-patch16-224")]
    [InlineData("CLIP-ViT-B-32-laion2B-s34B-b79K")]
    [InlineData("siglip-so400m-patch14-384")]
    public async Task DualEncoderMatchesTransformers(string name)
    {
        if (!Available(name) || !Reference(name).TryGetProperty("text_embeddings", out var textRef))
        {
            return;
        }

        var reference = Reference(name);
        var imageExpected = Floats(reference.GetProperty("image_embedding"));
        var texts = reference.GetProperty("texts").EnumerateArray().Select(t => t.GetString()!).ToArray();
        var textExpected = textRef.EnumerateArray().Select(Floats).ToArray();
        var logitsExpected = Floats(reference.GetProperty("logits_per_image"));
        var image = PngCodec.Read(Path.Combine(Model(name), "sharplight-sample.png"));
        foreach (var (backend, slot) in Backends())
        {
            using var model = Load(name, slot);
            Assert.Equal("image-embedding", model.Description.Kind);
            var ids = reference.GetProperty("token_ids").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
            var tokenizer = model.CreateTokenizer("tokenizer");
            for (int i = 0; i < texts.Length; i++)
            {
                var actual = tokenizer.Encode(texts[i]);
                Assert.True(ids[i].Take(actual.Length).SequenceEqual(actual), $"{name}: ids [{string.Join(", ", actual)}] vs reference [{string.Join(", ", ids[i].Take(actual.Length + 2))}]");
                Assert.True(ids[i].Skip(actual.Length).All(t => t == ids[i][^1]), $"{name}: ids [{string.Join(", ", actual)}] are followed by more than padding in [{string.Join(", ", ids[i].Take(actual.Length + 2))}]");
            }

            using var embedder = new TaskClientProvider([.. Env.TaskFactories()]).Create<IImageEmbedder>(model);
            Assert.True(embedder.SupportsText);
            var embedding = await embedder.EmbedImageAsync(image, new ImageEmbeddingOptions { Normalize = false }, TestContext.Current.CancellationToken);
            double imageRms = Env.Compare(embedding, imageExpected).Rms;
            TestContext.Current.SendDiagnosticMessage($"{name} {backend}: image embedding rms {imageRms:e2}");
            Assert.True(imageRms < 3e-3, $"{name} {backend}: image embedding rms {imageRms}");
            for (int i = 0; i < texts.Length; i++)
            {
                var text = await embedder.EmbedTextAsync(texts[i], new ImageEmbeddingOptions { Normalize = false }, TestContext.Current.CancellationToken);
                var (maxAbs, _, rms) = Env.Compare(text, textExpected[i]);
                TestContext.Current.SendDiagnosticMessage($"{name} {backend}: text {i} rms {rms:e2}");
                Assert.True(rms < 3e-3, $"{name} {backend}: text '{texts[i]}' relative rms {rms}, max abs {maxAbs}");
            }

            var zeroShot = await embedder.ClassifyZeroShotAsync(image, texts, new ImageClassificationOptions { TopK = 0 }, TestContext.Current.CancellationToken);
            var logits = new float[texts.Length];
            foreach (var p in zeroShot.Predictions)
            {
                logits[p.Index] = p.Logit;
            }

            var (zMax, _, zRms) = Env.Compare(logits, logitsExpected);
            TestContext.Current.SendDiagnosticMessage($"{name} {backend}: zero-shot logits rms {zRms:e2}");
            Assert.True(zRms < 5e-3, $"{name} {backend}: zero-shot logits rms {zRms}, max abs {zMax}");
            Assert.Equal(Array.IndexOf(logitsExpected, logitsExpected.Max()), zeroShot.Top!.Index);
        }
    }
}
