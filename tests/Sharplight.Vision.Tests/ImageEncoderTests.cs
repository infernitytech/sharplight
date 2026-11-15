using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
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

    private static string Model(string name) => Path.Combine(Home, "models/vision", name);

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

    [Fact]
    public async Task VitClassifierMatchesTransformers()
    {
        if (!Available("vit-base-patch16-224"))
        {
            return;
        }

        var reference = Reference("vit-base-patch16-224");
        var expected = Floats(reference.GetProperty("logits"));
        var image = PngCodec.Read(Path.Combine(Model("vit-base-patch16-224"), "sharplight-sample.png"));
        foreach (var (backend, slot) in Backends())
        {
            using var model = Load("vit-base-patch16-224", slot);
            Assert.Equal("image-classification", model.Description.Kind);
            using var classifier = new TaskClientProvider([.. Factories()]).Create<IImageClassifier>(model);
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

    [Fact]
    public async Task Dinov2EmbeddingMatchesTransformers()
    {
        if (!Available("dinov2-small"))
        {
            return;
        }

        var reference = Reference("dinov2-small");
        var expected = Floats(reference.GetProperty("pooled"));
        var image = PngCodec.Read(Path.Combine(Model("dinov2-small"), "sharplight-sample.png"));
        foreach (var (backend, slot) in Backends())
        {
            using var model = Load("dinov2-small", slot);
            Assert.Equal("image-embedding", model.Description.Kind);
            using var embedder = new TaskClientProvider([.. Factories()]).Create<IImageEmbedder>(model);
            var embedding = await embedder.EmbedImageAsync(image, new ImageEmbeddingOptions { Normalize = false });
            Assert.Equal(expected.Length, embedder.Dimensions);
            var (maxAbs, _, rms) = Env.Compare(embedding, expected);
            Assert.True(rms < 2e-3, $"{backend}: embedding relative rms {rms}, max abs {maxAbs}");
        }
    }

    // The factories registered by AddSharplightTasks that exist as instances (the image encoders').
    private static IEnumerable<ITaskClientFactory> Factories()
    {
        var services = new ServiceCollection();
        services.AddSharplightTasks();
        return services.Where(d => d.ServiceType == typeof(ITaskClientFactory) && d.ImplementationInstance is not null).Select(d => (ITaskClientFactory)d.ImplementationInstance!);
    }
}
