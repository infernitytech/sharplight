using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using Sharplight.Audio;
using Sharplight.Hub;
using Sharplight.ModelFeed.Index;
using Sharplight.ModelFeed.Resolution;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Feed;
using Sharplight.Studio.Services.Smoke;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Speech;

namespace Sharplight.Studio.Tests;

/// <summary>The <c>--smoke</c> run (<see cref="SmokeRunner"/>) over fakes: its steps, its failure reporting and the JSON report.</summary>
public sealed class SmokeRunnerTests
{
    [Fact]
    public async Task Run_PassesEveryStep_AndLoadsOnTheCpu()
    {
        using var dir = new TempDir();
        var runtime = new FakeRuntime();
        var feed = new SmokeFeed(dir.File("store"));
        var model = new FakeRecognizer();
        var runner = Runner(dir, feed, runtime, model);

        var report = await runner.RunAsync("1.2.3-alpha.gabc", "Portable at x", TestContext.Current.CancellationToken);

        Assert.True(report.Passed, report.Error);
        Assert.Equal(["devices", "feed", "install", "decode", "load", "infer", "unload"], report.Steps.Select(step => step.Name));
        Assert.Equal("whisper-tiny", report.Package);
        Assert.Equal(FakeRuntime.Cpu.Id, runtime.Requests.Single().Settings.DeviceId);
        Assert.Equal(Sharplight.Models.ModelKinds.SpeechRecognition, runtime.Requests.Single().Kind);
        Assert.Single(model.Requests);
        Assert.Empty(runtime.LoadedModels);
        Assert.Contains(report.Devices, device => device.Contains("cpu", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Run_StopsAtTheFirstFailingStep_AndReportsIt()
    {
        using var dir = new TempDir();
        var runtime = new FakeRuntime();
        var feed = new SmokeFeed(dir.File("store")) { FailInstall = true };
        var runner = Runner(dir, feed, runtime, new FakeRecognizer());

        var report = await runner.RunAsync("1.0.0", "System", TestContext.Current.CancellationToken);

        Assert.False(report.Passed);
        Assert.Equal(["devices", "feed", "install"], report.Steps.Select(step => step.Name));
        Assert.Contains("digest mismatch", report.Error, StringComparison.Ordinal);
        Assert.Empty(runtime.Requests);
    }

    [Fact]
    public async Task Run_FailsWithoutACpuDevice()
    {
        using var dir = new TempDir();
        var runtime = new FakeRuntime { Devices = [FakeRuntime.Gpu] };
        var runner = Runner(dir, new SmokeFeed(dir.File("store")), runtime, new FakeRecognizer());

        var report = await runner.RunAsync("1.0.0", "System", TestContext.Current.CancellationToken);

        Assert.False(report.Passed);
        Assert.Equal("devices", Assert.Single(report.Steps).Name);
    }

    [Fact]
    public async Task Report_SerializesWithThePassedFlagAndTheSteps()
    {
        using var dir = new TempDir();
        var runtime = new FakeRuntime();
        var runner = Runner(dir, new SmokeFeed(dir.File("store")), runtime, new FakeRecognizer());
        var report = await runner.RunAsync("1.0.0", "Portable", TestContext.Current.CancellationToken);

        using var json = JsonDocument.Parse(JsonSerializer.Serialize(report, StudioJsonContext.Default.SmokeReport));

        Assert.True(json.RootElement.GetProperty("Passed").GetBoolean());
        Assert.Equal("1.0.0", json.RootElement.GetProperty("Version").GetString());
        Assert.Equal(7, json.RootElement.GetProperty("Steps").GetArrayLength());
    }

    [Fact]
    public void Wav_IsAMono16kHzPcmFileOfOneAndAHalfSeconds()
    {
        var audio = new WavAudioDecoder().Decode(SmokeAudio.CreateWav());

        Assert.Equal(16000, audio.SampleRate);
        Assert.Equal(1, audio.Channels);
        Assert.Equal(24000, audio.Samples.Length);
        Assert.True(audio.Samples.Max() > 0.1f);
    }

    private static SmokeRunner Runner(TempDir dir, SmokeFeed feed, FakeRuntime runtime, FakeRecognizer vad)
    {
        runtime.Customize = created => created.Clients[typeof(ISpeechRecognizer)] = vad;
        return new SmokeRunner(feed, runtime, new WavAudioDecoder(), dir.File("work"), NullLogger.Instance);
    }

    private sealed class SmokeFeed(string store) : IFeedService
    {
        public bool FailInstall { get; init; }

        public string Url => FeedOptions.DefaultUrl;

        public string Channel => FeedOptions.Stable;

        public CompatibilityEnvironment Environment { get; } = new();

        public Task<FeedListing> GetListingAsync(CancellationToken cancellationToken) => Task.FromResult(new FeedListing
        {
            Verified = true,
            Packages = [new FeedPackageSummary { Id = "whisper-tiny", LatestVersion = "1.0.0", ManifestDigest = "m", IndexDigest = "i" }],
        });

        public Task<FeedInstallResult> InstallAsync(string id, IProgress<FeedInstallProgress>? progress, CancellationToken cancellationToken)
        {
            if (FailInstall)
            {
                throw new IOException("digest mismatch");
            }

            Directory.CreateDirectory(store);
            var manifest = new ModelManifest { Repository = "sharplight/whisper-tiny", Alias = "whisper-tiny", Commit = "0", FeedPackage = id, FeedVersion = "1.0.0", Directory = store };
            File.WriteAllBytes(manifest.PackagePath, [1, 2, 3]);
            return Task.FromResult(new FeedInstallResult([manifest], manifest));
        }
    }
}
