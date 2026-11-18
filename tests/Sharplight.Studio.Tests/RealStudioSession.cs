using Sharplight.Audio;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Audio;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Studio's services on the real engine runtime for the end-to-end tests: models load through <see cref="EngineModelRuntime"/> like in
/// the app (the kind comes from <see cref="ModelInspector"/>), the microphone and the speakers are the real recorder and player over
/// a fake device (the microphone delivers the samples given, the speaker keeps what was played).
/// </summary>
internal sealed class RealStudioSession : IDisposable
{
    private readonly TestServices _fakes = new();

    public RealStudioSession()
    {
        Audio = new FakeAudioDevicesAccess(_fakes);
        Runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(Runtime.Devices.Count == 0, "no compute device");
        var decoder = new WavAudioDecoder();
        Services = _fakes.Services with
        {
            Runtime = Runtime,
            Recorder = new EngineAudioRecorder(_fakes.AudioDevices, new WavAudioEncoder(), _fakes.Settings),
            Player = new EngineAudioPlayer(decoder, _fakes.AudioDevices, _fakes.Settings),
        };
    }

    /// <summary>The home directory the model mirror lives under.</summary>
    public static string Home { get; } = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    public EngineModelRuntime Runtime { get; }

    public StudioServices Services { get; }

    public FakeAudioDevicesAccess Audio { get; }

    public string Directory => _fakes.Directory;

    public Fakes.FakePlatform Platform => _fakes.Platform;

    /// <summary>The path below the model mirror (<c>~/models</c>); the test is skipped when it does not exist.</summary>
    public static string Require(string relative)
    {
        string path = System.IO.Path.Combine(Home, "models", relative);
        Assert.SkipUnless(File.Exists(path) || System.IO.Directory.Exists(path), $"{relative} not found");
        return path;
    }

    /// <summary>Waits until <paramref name="condition"/> holds (the dispatcher and the playback run on other threads).</summary>
    public static async Task WaitFor(Func<bool> condition, int seconds = 120)
    {
        for (int i = 0; i < seconds * 20 && !condition(); i++)
        {
            await Task.Delay(50, TestContext.Current.CancellationToken);
        }

        Assert.True(condition());
    }

    /// <summary>Whether the samples hold more than silence.</summary>
    public static bool IsAudible(IEnumerable<float> samples) => samples.Any(sample => MathF.Abs(sample) > 0.02f);

    /// <summary>Runs the page's queue item to the end and returns it, failing with its detail when it did not complete.</summary>
    public static async Task<RunItemViewModel> FinishAsync(InferencePageViewModel page, int minutes = 5)
    {
        var item = page.Queue.Items[^1];
        await item.Completion.WaitAsync(TimeSpan.FromMinutes(minutes), TestContext.Current.CancellationToken);
        Assert.True(item.Status == RunStatus.Completed, item.Detail + item.StatusText);
        return item;
    }

    /// <summary>Loads a model the way the library's load panel does: the kind is what the inspector recognizes.</summary>
    public async Task<ILoadedModel> LoadAsync(string id, string path, string? expectedKind = null, CancellationToken cancellationToken = default)
    {
        string kind = ModelInspector.Inspect(path).Kind;
        if (expectedKind is not null)
        {
            Assert.Equal(expectedKind, kind);
        }

        var settings = new LoadSettings { DeviceId = Runtime.Devices[0].Id, ContextLength = 2048, BatchTokens = 256 };
        return await Runtime.LoadAsync(new ModelLoadRequest(id, path, settings, [], Kind: kind), null, cancellationToken);
    }

    public void Dispose()
    {
        Runtime.Dispose();
        _fakes.Dispose();
    }

    /// <summary>The fake audio device: what the microphone delivers and what the speaker received.</summary>
    internal sealed class FakeAudioDevicesAccess(TestServices services)
    {
        public float[] Microphone
        {
            set => services.AudioDevices.Microphone = value;
        }

        public List<float> Played => services.AudioDevices.Output?.Written ?? [];

        public int PlayedSampleRate => services.AudioDevices.LastOptions?.SampleRate ?? 0;
    }
}
