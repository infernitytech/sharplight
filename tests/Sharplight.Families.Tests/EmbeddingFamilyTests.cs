using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm.Embeddings;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Bidirectional (embedding) families against reference embeddings stored next to the model as <c>reference_embeddings.json</c>
/// (<c>inputs</c>: token-id sequences, <c>embeddings</c>: the expected pooled vectors before normalization, <c>pooling</c>). Tiny Hugging Face
/// checkpoints get theirs from transformers (<c>tools/mk_tiny_encoders.py</c>); tiny GGUF files (<c>tools/mk_tiny_gguf.py</c>, references named
/// <c>&lt;file&gt;.reference_embeddings.json</c>) from llama.cpp b11177's llama-server (<c>tools/llamacpp_embeddings.py</c>). Every case runs on the
/// CPU provider and on the first Vulkan device, without a CPU fallback, and skips when its files are missing.
/// </summary>
public class EmbeddingFamilyTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    static EmbeddingFamilyTests()
    {
        GgufImporter.Register();
        HuggingFaceImporter.Register();
    }

    /// <summary>Model path under ~/models and a description.</summary>
    public static TheoryData<string, string, bool> Cases()
    {
        var data = new TheoryData<string, string, bool>();
        string[] models =
        [
            // transformers references.
            "hf/tiny-random-eurobert",
            "hf/tiny-random-embeddinggemma",
            "hf/tiny-random-modernbert",
            // T5 encoders (mk_tiny_t5.py): relative position bias, the 150-token input beyond the bias range.
            "hf/tiny-random-t5",
            "hf/tiny-random-t5-v1_1",
            // llama.cpp references.
            "gguf/tiny-random-eurobert-f32.gguf",
            "gguf/tiny-random-neo-bert-f32.gguf",
            "gguf/tiny-random-llama-embed-f32.gguf",
            "gguf/tiny-random-gemma-embedding-f32.gguf",
            "gguf/tiny-random-modern-bert-f32.gguf",
            "gguf/tiny-random-jina-bert-v2-f32.gguf",
            // Real checkpoints: ModernBERT-base and EuroBERT-210m (transformers, mk_tiny_encoders.py --reference), jina-embeddings-v2-small-en
            // converted with convert_hf_to_gguf.py (llama.cpp).
            "hf/ModernBERT-base",
            "hf/EuroBERT-210m",
            "gguf/real-embed/jina-embeddings-v2-small-en-f32.gguf",
        ];
        foreach (string model in models)
        {
            data.Add(model, model, false);
            data.Add(model, model, true);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesReference(string model, string name, bool vulkan)
    {
        _ = name;
        string path = Path.Combine(Home, "models", model);
        string referencePath = Directory.Exists(path) ? Path.Combine(path, "reference_embeddings.json") : path + ".reference_embeddings.json";
        if (!File.Exists(referencePath))
        {
            Assert.Skip($"{model} or its reference embeddings are missing");
        }

        using var json = JsonDocument.Parse(File.ReadAllText(referencePath));
        var root = json.RootElement;
        var inputs = root.GetProperty("inputs").EnumerateArray().Select(a => (IReadOnlyList<int>)[.. a.EnumerateArray().Select(e => e.GetInt32())]).ToList();
        var expected = root.GetProperty("embeddings").EnumerateArray().Select(a => a.EnumerateArray().Select(e => (float)e.GetDouble()).ToArray()).ToList();
        string pooling = root.GetProperty("pooling").GetString()!;

        using var embedder = EmbeddingModel.Load(path, new EmbeddingLoadOptions
        {
            Devices = [Device(vulkan)], Pooling = pooling, Normalize = false, MaxBatchTokens = 256, MaxBatchSequences = 8,
        });
        var result = embedder.EmbedTokens(inputs);
        for (int i = 0; i < inputs.Count; i++)
        {
            var got = result.Vectors[i];
            Assert.Equal(expected[i].Length, got.Length);
            double scale = expected[i].Max(MathF.Abs);
            double maxDiff = expected[i].Zip(got, (a, b) => (double)MathF.Abs(a - b)).Max();
            // fp32 CPU; the Vulkan flash-attention kernel runs Q, K and P in f16.
            double tolerance = (vulkan ? 2e-2 : 2e-3) * scale;
            Assert.True(maxDiff <= tolerance, $"{model} input {i}: max |diff| {maxDiff:G4} (largest |value| {scale:G4})");
        }
    }

    private static DeviceSlot Device(bool vulkan)
    {
        if (!vulkan)
        {
            var cpu = new CpuExecutionProvider();
            return new DeviceSlot { Device = cpu.OpenDevice(cpu.GetDevices()[0]) };
        }

        var provider = new VulkanProvider();
        if (!provider.CheckAvailability().IsAvailable)
        {
            Assert.Skip("Vulkan is not available");
        }

        var devices = provider.GetDevices();
        var device = Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_VK_DEVICE") is { } e && int.TryParse(e, out int index) && index < devices.Count ? devices[index]
            : devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices.FirstOrDefault();
        if (device is null)
        {
            Assert.Skip("no Vulkan device");
        }

        return new DeviceSlot { Device = provider.OpenDevice(device!) };
    }
}
