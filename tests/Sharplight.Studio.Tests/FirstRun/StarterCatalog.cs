using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Index;

namespace Sharplight.Studio.Tests.FirstRun;

/// <summary>Feed package summaries shaped like the real feed's: a family of chat models of growing size, speech to text with its voice detector, embeddings.</summary>
internal static class StarterCatalog
{
    public const long Gib = 1L << 30;

    public static FeedPackageSummary Chat(string id, string family, double memoryGib, double downloadGib, string[]? tasks = null, PackageState? state = null)
        => Make(id, family, "text-generation", tasks ?? ["chat", "completion"], memoryGib, downloadGib, state: state);

    public static FeedPackageSummary Speech(string id, double memoryGib, double downloadGib, params string[] dependencies)
        => Make(id, "Whisper", "speech-recognition", ["transcription"], memoryGib, downloadGib, dependencies);

    public static FeedPackageSummary Embedding(string id, double memoryGib, double downloadGib)
        => Make(id, "MiniLM", "embeddings", ["embedding"], memoryGib, downloadGib);

    public static FeedPackageSummary Detector(string id) => Make(id, "Silero VAD", "voice-activity", ["voice-activity"], 0.4, 0.002);

    /// <summary>Qwen3 in five sizes, Gemma, whisper with its detector, and an embedding model.</summary>
    public static IReadOnlyList<FeedPackageSummary> Feed() =>
    [
        Chat("qwen3-0.6b.q8-0-gguf", "Qwen3", 1.1, 0.6),
        Chat("qwen3-1.7b.q8-0-gguf", "Qwen3", 2.2, 1.8),
        Chat("qwen3-8b.q4-k-m-gguf", "Qwen3", 5.8, 5.0),
        Chat("qwen3-14b.q4-k-m-gguf", "Qwen3", 10.0, 9.0),
        Chat("qwen3-32b.q4-k-m-gguf", "Qwen3", 21.5, 19.5),
        Chat("gemma-3-27b.q4-k-m-gguf", "Gemma", 18.0, 16.0),
        Speech("whisper-tiny.q5-1-whispercpp", 0.45, 0.04, "silero-vad-v5.onnx"),
        Detector("silero-vad-v5.onnx"),
        Embedding("all-minilm-l6-v2.f32-safetensors", 0.5, 0.09),
    ];

    private static FeedPackageSummary Make(string id, string family, string kind, string[] tasks, double memoryGib, double downloadGib, string[]? dependencies = null,
        PackageState? state = null)
        => new()
        {
            Id = id,
            LatestVersion = "1.0.0",
            Versions = ["1.0.0"],
            Display = new PackageDisplay { Name = $"{family} {id}", Family = family, Variant = "test" },
            Model = new PackageModel { Kind = kind, Tasks = tasks, Architecture = family.ToLowerInvariant() },
            Requirements = new PackageRequirements
            {
                Memory = [new MemoryRequirement { Device = "gpu", Weights = (long)(memoryGib * Gib) },
                    new MemoryRequirement { Device = "cpu", Weights = (long)(memoryGib * Gib) }],
            },
            Dependencies = [.. (dependencies ?? []).Select(dependency => new PackageDependency { Id = dependency, Range = "[1.0.0,2.0.0)" })],
            Size = (long)(downloadGib * Gib),
            State = state,
            ManifestDigest = "sha256:" + new string('0', 64),
            IndexDigest = "sha256:" + new string('1', 64),
        };
}
