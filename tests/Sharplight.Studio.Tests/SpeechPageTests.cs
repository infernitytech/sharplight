using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Studio.Models;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Speech;

namespace Sharplight.Studio.Tests;

/// <summary>The Transcribe and Speak pages over fake task clients.</summary>
public sealed class SpeechPageTests
{
    private static string WriteWav(TestServices services, string name, double seconds = 3)
    {
        string path = services.Directory + "/" + name;
        int rate = 16000;
        WavFile.Write(path, new AudioBuffer([.. Enumerable.Range(0, (int)(rate * seconds)).Select(i => 0.2f * MathF.Sin(2 * MathF.PI * 220 * i / rate))], rate));
        return path;
    }

    private static FakeModel Recognizer(TestServices services, out FakeRecognizer client)
    {
        client = new FakeRecognizer();
        var model = new FakeModel("whisper-tiny") { Kind = ModelKinds.SpeechRecognition };
        model.Clients[typeof(ISpeechRecognizer)] = client;
        services.Runtime.Add(model);
        return model;
    }

    private static FakeModel Synthesizer(TestServices services, out FakeSynthesizer client)
    {
        client = new FakeSynthesizer();
        var model = new FakeModel("outetts") { Kind = ModelKinds.SpeechSynthesis };
        model.Clients[typeof(ISpeechSynthesizer)] = client;
        services.Runtime.Add(model);
        return model;
    }

    private static async Task WaitFor(Func<bool> condition)
    {
        for (int i = 0; i < 400 && !condition(); i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        Assert.True(condition());
    }

    [Fact]
    public async Task Transcribe_StreamsSegments_AndKeepsTheRunInHistoryAndGallery()
    {
        using var services = new TestServices();
        Recognizer(services, out var client);
        var page = new TranscribePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        Assert.True(page.HasModel);
        Assert.False(page.RunCommand.CanExecute(null));
        Assert.Contains("Choose or record audio", page.RunHint, StringComparison.Ordinal);

        Assert.True(await page.SetAudioAsync(WriteWav(services, "talk.wav")));
        Assert.Contains("3.0 s", page.AudioSummary, StringComparison.Ordinal);
        Assert.True(page.RunCommand.CanExecute(null));

        page.Panel.Parameters.OfType<Sharplight.Studio.ViewModels.Inference.Parameters.NumberParameterViewModel>().Single().Value = 5;
        page.Panel.Parameters.OfType<Sharplight.Studio.ViewModels.Inference.Parameters.ChoiceParameterViewModel>().First(parameter => parameter.Key == "task").Deserialize("translate");
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        Assert.Equal(RunStatus.Completed, page.Queue.Items[0].Status);
        Assert.Equal(2, page.Segments.Count);
        Assert.Equal("00:00.0 - 00:02.5", page.Segments[0].Time);
        Assert.Equal("And so, my fellow Americans, ask not what your country can do for you.", page.TranscriptText);
        Assert.Equal("en", page.ResultLanguage);
        Assert.True(page.HasResult);
        var options = Assert.Single(client.Requests)!;
        Assert.Equal(5, options.BeamSize);
        Assert.True(options.Translate);
        Assert.Null(options.Language);

        var run = Assert.Single(page.History.Items);
        Assert.Equal("talk.wav", run.Title);
        Assert.Equal("whisper-tiny", run.Record.Model);
        var output = Assert.Single(page.Gallery.Items);
        Assert.Equal(["transcript.txt", "transcript.srt", "transcript.vtt", "transcript.json"], output.Files.Select(file => file.Name));
        Assert.Contains("language: en", output.Facts);
        string srt = await File.ReadAllTextAsync(output.Files[1].Path, TestContext.Current.CancellationToken);
        Assert.StartsWith("1\n00:00:00,000 --> 00:00:02,500\nAnd so, my fellow Americans,\n\n2\n", srt, StringComparison.Ordinal);
        string vtt = await File.ReadAllTextAsync(output.Files[2].Path, TestContext.Current.CancellationToken);
        Assert.StartsWith("WEBVTT\n\n00:00:00.000 --> 00:00:02.500\n", vtt, StringComparison.Ordinal);

        // The parameters of the last run come back when the page is opened again, and a history entry restores its result and options.
        var reopened = new TranscribePageViewModel(services.Services, new FakeNavigator());
        Assert.Equal(5, reopened.Panel.Options.BeamSize);
        reopened.Panel.ResetToDefaultsCommand.Execute(null);
        reopened.History.RestoreCommand.Execute(reopened.History.Items[0]);
        Assert.Equal(2, reopened.Segments.Count);
        Assert.Equal(5, reopened.Panel.Options.BeamSize);
        Assert.Equal("talk.wav", reopened.AudioPath);
    }

    [Fact]
    public async Task Transcribe_ExportsTextSrtAndVtt_AndCopies()
    {
        using var services = new TestServices();
        Recognizer(services, out _);
        var page = new TranscribePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await page.SetAudioAsync(WriteWav(services, "speech.wav"));
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        string target = services.Directory + "/out.srt";
        services.Platform.SaveTo = target;
        await page.ExportCommand.ExecuteAsync("srt");
        Assert.Contains("00:00:02,500 --> 00:00:06,000", await File.ReadAllTextAsync(target, TestContext.Current.CancellationToken), StringComparison.Ordinal);
        await page.ExportCommand.ExecuteAsync("txt");
        Assert.Equal("And so, my fellow Americans, ask not what your country can do for you.\n", await File.ReadAllTextAsync(target, TestContext.Current.CancellationToken));

        await page.CopyCommand.ExecuteAsync(null);
        Assert.Equal("And so, my fellow Americans, ask not what your country can do for you.", services.Platform.Copied.Single());
    }

    [Fact]
    public async Task Transcribe_Cancel_StopsTheRun_AndAFailureIsShown()
    {
        using var services = new TestServices();
        Recognizer(services, out var client);
        client.Gate = new TaskCompletionSource();
        var page = new TranscribePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await page.SetAudioAsync(WriteWav(services, "long.wav"));
        page.RunCommand.Execute(null);
        var item = page.Queue.Items.Single();
        await WaitFor(() => page.Segments.Count == 1);
        Assert.True(page.IsTranscribing);

        page.CancelCommand.Execute(null);
        await item.Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(RunStatus.Cancelled, item.Status);
        Assert.False(page.IsTranscribing);
        Assert.Empty(page.History.Items);   // a cancelled run is not kept

        client.Gate = null;
        client.Failure = new InvalidOperationException("decoder exploded");
        page.RunCommand.Execute(null);
        var failed = page.Queue.Items[1];
        await failed.Completion.WaitAsync(TestContext.Current.CancellationToken);
        Assert.Equal(RunStatus.Failed, failed.Status);
        Assert.Equal("decoder exploded", page.Notice);
        var run = Assert.Single(page.History.Items);
        Assert.True(run.IsFailed);
        Assert.Empty(page.Gallery.Items);
    }

    [Fact]
    public async Task Transcribe_RefusesUnreadableFiles_AndMissingFiles()
    {
        using var services = new TestServices();
        Recognizer(services, out _);
        var page = new TranscribePageViewModel(services.Services, new FakeNavigator());
        string mp3 = services.Directory + "/song.mp3";
        await File.WriteAllBytesAsync(mp3, [1, 2, 3], TestContext.Current.CancellationToken);

        Assert.False(await page.SetAudioAsync(mp3));
        Assert.Contains("cannot be read", page.Notice, StringComparison.Ordinal);
        Assert.False(await page.SetAudioAsync(services.Directory + "/missing.wav"));
        Assert.Contains("does not exist", page.Notice, StringComparison.Ordinal);
        string broken = services.Directory + "/broken.wav";
        await File.WriteAllBytesAsync(broken, new byte[64], TestContext.Current.CancellationToken);
        Assert.False(await page.SetAudioAsync(broken));
        Assert.Contains("cannot be read", page.Notice, StringComparison.Ordinal);
        Assert.False(page.HasAudio);
    }

    [Fact]
    public async Task Transcribe_RecordsTheMicrophone_WhenARecorderExists()
    {
        using var services = new TestServices();
        Recognizer(services, out _);
        services.Audio.OnRecorded = path => WavFile.Write(path, new AudioBuffer(new float[16000], 16000));
        var page = new TranscribePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        Assert.True(page.CanRecord);

        await page.ToggleRecordingCommand.ExecuteAsync(null);
        Assert.True(page.IsRecording);
        Assert.False(page.RunCommand.CanExecute(null));
        await page.ToggleRecordingCommand.ExecuteAsync(null);
        Assert.False(page.IsRecording);
        Assert.True(page.HasAudio);
        Assert.True(page.RunCommand.CanExecute(null));

        services.Audio.IsAvailable = false;
        var noMic = new TranscribePageViewModel(services.Services, new FakeNavigator());
        Assert.False(noMic.CanRecord);
        await noMic.ToggleRecordingCommand.ExecuteAsync(null);
        Assert.Contains("No audio device", noMic.Notice, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Speak_Synthesizes_AndPlaysSavesAndKeepsTheAudio()
    {
        using var services = new TestServices();
        Synthesizer(services, out var client);
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        Assert.False(page.RunCommand.CanExecute(null));
        Assert.Contains("Type the text", page.RunHint, StringComparison.Ordinal);

        page.InputText = "Hello from Sharplight.";
        Assert.True(page.RunCommand.CanExecute(null));
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        await WaitFor(() => page.HasResult);

        var (text, options, streaming) = Assert.Single(client.Requests);
        Assert.Equal("Hello from Sharplight.", text);
        Assert.False(streaming);
        Assert.Equal(0.4f, options!.Sampling.Temperature);
        Assert.Null(options.Voice);
        Assert.Contains("audio: 1.0 s", page.ResultSummary, StringComparison.Ordinal);
        var output = Assert.Single(page.Gallery.Items);
        Assert.Equal("speech.wav", output.Files[0].Name);
        var stored = WavFile.Read(output.Files[0].Path);
        Assert.Equal(24000, stored.SampleRate);
        Assert.Equal(24000, stored.FrameCount);

        await page.PlayCommand.ExecuteAsync(null);
        Assert.Equal([output.Files[0].Path], services.Audio.Played);
        Assert.False(page.IsPlaying);

        services.Platform.SaveTo = services.Directory + "/saved.wav";
        await page.SaveCommand.ExecuteAsync(null);
        Assert.Equal(24000, WavFile.Read(services.Platform.SaveTo).SampleRate);

        // Restoring a history entry shows its text, options and audio again.
        var other = new SpeakPageViewModel(services.Services, new FakeNavigator());
        other.History.RestoreCommand.Execute(other.History.Items[0]);
        Assert.Equal("Hello from Sharplight.", other.InputText);
        Assert.True(other.HasResult);
    }

    [Fact]
    public async Task Speak_OffersTheVoicesOfTheModel_AndStreamsWhilePlaying()
    {
        using var services = new TestServices();
        Synthesizer(services, out var client);
        client.Voices = ["female_1", "male_1"];
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        await WaitFor(() => page.Panel.Parameters.First(parameter => parameter.Key == "voice").IsVisibleNow);

        var voice = page.Panel.Parameters.OfType<Sharplight.Studio.ViewModels.Inference.Parameters.ChoiceParameterViewModel>().First(parameter => parameter.Key == "voice");
        Assert.Equal(["Default voice", "female_1", "male_1"], voice.Choices.Select(choice => choice.Label));
        voice.Deserialize("male_1");
        page.Panel.Parameters.OfType<Sharplight.Studio.ViewModels.Inference.Parameters.BoolParameterViewModel>().Single().Value = true;
        page.InputText = "one two three";
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);

        var (_, options, streaming) = Assert.Single(client.Requests);
        Assert.True(streaming);
        Assert.Equal("male_1", options!.Voice);
        Assert.Equal(3, services.Audio.Played.Count);   // every streamed segment was played
        Assert.Contains("voice: male_1", page.Gallery.Items[0].Facts);
    }

    [Fact]
    public async Task Speak_WithoutAPlayer_SaysSoAndOpensTheFile()
    {
        using var services = new TestServices();
        Synthesizer(services, out _);
        services.Audio.IsAvailable = false;
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        Assert.False(page.CanPlay);
        page.InputText = "hi";
        page.RunCommand.Execute(null);
        await page.Queue.Items.Single().Completion.WaitAsync(TestContext.Current.CancellationToken);
        await WaitFor(() => page.HasResult);

        await page.PlayCommand.ExecuteAsync(null);
        Assert.Contains("No audio device", page.Notice, StringComparison.Ordinal);
        Assert.Single(services.Platform.Opened);
        Assert.Empty(services.Audio.Played);
    }

    [Fact]
    public async Task Pages_OpenFromTheNavigation_AndFollowTheResidentModels()
    {
        using var services = new TestServices();
        var main = new MainWindowViewModel(services.Services);
        var model = Recognizer(services, out _);

        Assert.True(main.OpenKind(ModelKinds.SpeechRecognition, model));
        await WaitFor(() => ReferenceEquals(main.CurrentPage, main.Transcribe));
        Assert.Equal("whisper-tiny", main.Transcribe.SelectedModel?.Id);
        Assert.False(main.Speak.HasModels);

        Assert.True(main.OpenKind(ModelKinds.SpeechSynthesis, null));
        await WaitFor(() => ReferenceEquals(main.CurrentPage, main.Speak));
        Assert.False(main.OpenKind("video-generation", null));
        Assert.Contains(main.Pages, page => page is TranscribePageViewModel);
        Assert.Contains(main.Pages, page => page is SpeakPageViewModel);

        await services.Runtime.UnloadAsync("whisper-tiny");
        Assert.False(main.Transcribe.HasModels);
        Assert.False(main.Transcribe.RunCommand.CanExecute(null));
    }

    [Fact]
    public void SpeechKinds_AreRunnable_AndTheirCardsOpenThePages()
    {
        Assert.True(ModelKindCatalog.For(ModelKinds.SpeechRecognition).Runnable);
        Assert.True(ModelKindCatalog.For(ModelKinds.SpeechSynthesis).Runnable);
        Assert.False(ModelKindCatalog.For("video-generation").Runnable);

        using var services = new TestServices();
        var navigator = new FakeNavigator();
        var tasks = new ViewModels.Tasks.TasksPageViewModel(services.Services, navigator);
        var speech = tasks.Kinds.Single(card => card.Info.Id == ModelKinds.SpeechRecognition);
        Assert.Equal("Open transcribe", speech.OpenText);
        speech.OpenCommand.Execute(null);
        tasks.Kinds.Single(card => card.Info.Id == ModelKinds.SpeechSynthesis).OpenCommand.Execute(null);
        Assert.Equal(["page:speech-recognition", "page:speech-synthesis"], navigator.Visited);
    }

    [Fact]
    public void TranscriptFile_RendersTextSrtAndVtt()
    {
        var file = new TranscriptFile
        {
            Segments = [new(0, 1.234, " Hello "), new(3661.5, 3662, "World")],
        };
        Assert.Equal("Hello World", file.ToText());
        Assert.Equal("1\n00:00:00,000 --> 00:00:01,234\nHello\n\n2\n01:01:01,500 --> 01:01:02,000\nWorld\n\n", file.ToSrt());
        Assert.Equal("WEBVTT\n\n00:00:00.000 --> 00:00:01.234\nHello\n\n01:01:01.500 --> 01:01:02.000\nWorld\n\n", file.ToVtt());
        Assert.Equal("Only text", new TranscriptFile { Text = " Only text " }.ToText());
    }
}
