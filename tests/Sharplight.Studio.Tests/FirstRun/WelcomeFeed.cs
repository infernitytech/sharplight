using Sharplight.ModelFeed;

namespace Sharplight.Studio.Tests.FirstRun;

/// <summary>Publishes the starter packages (chat, speech with its voice detector, embeddings) into a local test feed.</summary>
internal static class WelcomeFeed
{
    public const string Chat = "starter-chat.q8";

    public const string Speech = "starter-speech.q5";

    public const string Detector = "starter-vad";

    public const string Embedding = "starter-embed.f32";

    public static async Task PublishAsync(FeedCatalogTests.Rig rig, long chatMemory = 2L << 30)
    {
        var chat = await rig.Feed.StageAsync(Chat, "1.0.0", 30_000, customize: manifest => Describe(manifest, "text-generation", ["chat"], "Starter Chat", "Qwen", chatMemory));
        var vad = await rig.Feed.StageAsync(Detector, "1.0.0", 5_000, customize: manifest => Describe(manifest, "voice-activity", ["voice-activity"], "Starter VAD", "Silero", 400L << 20));
        var speech = await rig.Feed.StageAsync(Speech, "1.0.0", 12_000, [new PackageDependency { Id = Detector, Range = "[1.0.0,2.0.0)" }],
            customize: manifest => Describe(manifest, "speech-recognition", ["transcription"], "Starter Speech", "Whisper", 450L << 20));
        var embedding = await rig.Feed.StageAsync(Embedding, "1.0.0", 8_000, customize: manifest => Describe(manifest, "embeddings", ["embedding"], "Starter Embeddings", "MiniLM", 500L << 20));
        await rig.Feed.Publisher.PublishAsync(rig.Feed.Folder, [chat, vad, speech, embedding], rig.Feed.FeedKey, new Sharplight.ModelFeed.Publishing.FeedPublishOptions { Channel = FeedChannels.Stable },
            TestContext.Current.CancellationToken);
    }

    private static PackageManifest Describe(PackageManifest manifest, string kind, string[] tasks, string name, string family, long memory) => manifest with
    {
        Display = new PackageDisplay { Name = name, Family = family, Variant = "test", Summary = $"{name} for tests." },
        Model = new PackageModel { Kind = kind, Tasks = tasks, Architecture = family.ToLowerInvariant() },
        Requirements = (manifest.Requirements ?? new PackageRequirements()) with { Memory = [new MemoryRequirement { Device = "gpu", Weights = memory }] },
    };
}
