using System.Collections.Specialized;
using Sharplight.Audio;
using Sharplight.Hub;
using Sharplight.ImageGeneration;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Chat;
using Sharplight.Studio.ViewModels.Image;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.ViewModels.Speech;
using Sharplight.Studio.ViewModels.Vision;
using Sharplight.Vision;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The UI thread stays responsive while every page works: the fakes block their thread for <see cref="StepWork"/> per step (as a graph
/// run does), a heartbeat on the real (headless) Avalonia UI thread must never be late by more than <see cref="UiHeartbeat.Tolerance"/>,
/// cancelling mid-run ends the run, and chatty progress reaches the UI thread throttled instead of once per report.
/// </summary>
public sealed class ResponsivenessTests
{
    private static readonly TimeSpan StepWork = TimeSpan.FromMilliseconds(600);

    /// <summary>The most UI jobs a run may post: about 15 a second plus a few for the start and the end.</summary>
    private static int PostBudget(TimeSpan elapsed) => (int)(elapsed.TotalSeconds * 20) + 30;

    [Fact]
    public Task Chat_StreamsOffTheUiThread_AndStops() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new CountingDispatcher());
        var model = services.Runtime.Add(new FakeModel("qwen3", ModelCapabilities.Chat | ModelCapabilities.Reasoning) { StepWork = StepWork });
        var chat = new ConversationViewModel(new Conversation(), services.Services, () => model, _ => { });

        var heartbeat = UiHeartbeat.Start();
        chat.InputText = "Hello";
        await chat.SendCommand.ExecuteAsync(null);
        heartbeat.StopAndAssertResponsive("chat");
        Assert.Equal("Hello there!", chat.Messages[^1].Content);

        // Stop while a step runs: the answer ends without the remaining steps.
        model.Script.Clear();
        model.Script.AddRange([.. Enumerable.Range(0, 20).Select(i => ChatStreamUpdate.Content($"word{i} "))]);
        chat.InputText = "Again";
        var sending = chat.SendCommand.ExecuteAsync(null);
        await WaitForAsync(() => chat.Messages.Count == 4 && chat.Messages[^1].Content.Length > 0);
        chat.StopCommand.Execute(null);
        await sending.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Assert.False(chat.IsGenerating);
        Assert.DoesNotContain("word19", chat.Messages[^1].Content, StringComparison.Ordinal);
    });

    [Fact]
    public Task Chat_ManyTokens_AreShownInBatches() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new CountingDispatcher());
        var model = services.Runtime.Add(new FakeModel("qwen3") { StepWork = TimeSpan.FromMilliseconds(2) });
        model.Script.Clear();
        model.Script.AddRange([.. Enumerable.Range(0, 400).Select(i => ChatStreamUpdate.Content("t "))]);
        var chat = new ConversationViewModel(new Conversation(), services.Services, () => model, _ => { });
        int contentChanges = 0;
        chat.Messages.CollectionChanged += (_, args) =>
        {
            if (args.Action == NotifyCollectionChangedAction.Add && args.NewItems?[0] is MessageViewModel { IsAssistant: true } message)
            {
                message.PropertyChanged += (_, change) =>
                {
                    if (change.PropertyName == nameof(MessageViewModel.Content))
                    {
                        contentChanges++;
                    }
                };
            }
        };

        var watch = System.Diagnostics.Stopwatch.StartNew();
        chat.InputText = "Count";
        await chat.SendCommand.ExecuteAsync(null);
        Assert.Equal(string.Concat(Enumerable.Repeat("t ", 400)), chat.Messages[^1].Content);
        Assert.InRange(contentChanges, 1, PostBudget(watch.Elapsed));
        Assert.True(contentChanges < 400, $"{contentChanges} content updates for 400 tokens");
    });

    [Fact]
    public Task Image_GeneratesOffTheUiThread_ThrottlesSteps_AndCancels() => HeadlessApp.RunAsync(async () =>
    {
        var dispatcher = new CountingDispatcher();
        using var services = new TestServices(dispatcher: dispatcher);
        var client = new FakeImageGenerator { StepWork = StepWork };
        var model = new FakeModel("sd") { Kind = ModelKinds.ImageGeneration };
        model.Clients[typeof(IImageGenerator)] = client;
        services.Runtime.Add(model);
        var page = new ImagePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.Prompt = "a lighthouse";
        Number(page.Panel.Parameters, "steps").Value = 3;

        await RunResponsiveAsync(page, page.RunCommand, "image generation");
        Assert.Single(page.Images);

        // Many fast steps with a preview each: the step text and previews reach the UI thread throttled.
        client.StepWork = TimeSpan.FromMilliseconds(3);
        Number(page.Panel.Parameters, "steps").Value = 100;
        dispatcher.Reset();
        var watch = System.Diagnostics.Stopwatch.StartNew();
        page.RunCommand.Execute(null);
        await page.Queue.Items[^1].Completion.WaitAsync(TestContext.Current.CancellationToken);
        await WaitForAsync(() => !page.Queue.Items[^1].IsActive);
        Assert.Equal(RunStatus.Completed, page.Queue.Items[^1].Status);
        Assert.True(dispatcher.Posts <= PostBudget(watch.Elapsed), $"{dispatcher.Posts} UI jobs for 100 steps in {watch.Elapsed.TotalSeconds:F1} s");

        client.StepWork = StepWork;
        Number(page.Panel.Parameters, "steps").Value = 20;
        await CancelMidRunAsync(page, page.RunCommand);
    });

    [Fact]
    public Task Transcribe_RunsOffTheUiThread_AndCancels() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new CountingDispatcher());
        var recognizer = new FakeRecognizer { StepWork = StepWork };
        var model = new FakeModel("whisper") { Kind = ModelKinds.SpeechRecognition };
        model.Clients[typeof(ISpeechRecognizer)] = recognizer;
        services.Runtime.Add(model);
        var page = new TranscribePageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        string wav = Path.Combine(services.Directory, "talk.wav");
        WavFile.Write(wav, new AudioBuffer(new float[16000 * 3], 16000));
        Assert.True(await page.SetAudioAsync(wav));

        await RunResponsiveAsync(page, page.RunCommand, "transcription");
        Assert.Equal(recognizer.Script.Count, page.Segments.Count);

        recognizer.Script.AddRange([.. Enumerable.Range(0, 20).Select(i => new TranscriptionSegment { Start = TimeSpan.FromSeconds(i), End = TimeSpan.FromSeconds(i + 1), Text = $"more {i}" })]);
        await CancelMidRunAsync(page, page.RunCommand);
    });

    [Fact]
    public Task Speak_RunsOffTheUiThread_AndCancels() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new CountingDispatcher());
        var synthesizer = new FakeSynthesizer { StepWork = StepWork };
        var model = new FakeModel("outetts") { Kind = ModelKinds.SpeechSynthesis };
        model.Clients[typeof(ISpeechSynthesizer)] = synthesizer;
        services.Runtime.Add(model);
        var page = new SpeakPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        page.InputText = "one two three";
        page.Panel.Parameters.OfType<BoolParameterViewModel>().Single(parameter => parameter.Key == "stream").Value = true;

        await RunResponsiveAsync(page, page.RunCommand, "speech synthesis");
        page.InputText = string.Join(' ', Enumerable.Range(0, 30).Select(i => $"word{i}"));
        await CancelMidRunAsync(page, page.RunCommand);
    });

    [Fact]
    public Task Vision_RunsOffTheUiThread() => HeadlessApp.RunAsync(async () =>
    {
        using var services = new TestServices(dispatcher: new CountingDispatcher());
        var vision = new FakeVision { StepWork = StepWork * 2 };
        var model = new FakeModel("vit") { Kind = ModelKinds.ImageClassification };
        model.Clients[typeof(IImageClassifier)] = vision;
        services.Runtime.Add(model);
        var page = new VisionPageViewModel(services.Services, new FakeNavigator());
        await page.ActivateAsync();
        string path = Path.Combine(services.Directory, "cat.png");
        PngCodec.Write(path, FakeImageGenerator.Gradient(80, 64, 2));
        Assert.True(await page.Source.SetFileAsync(path));

        await RunResponsiveAsync(page, page.RunCommand, "image classification");
    });

    [Fact]
    public Task Library_PrecompileAndConvert_StayOffTheUiThread_AndThrottleProgress() => HeadlessApp.RunAsync(async () =>
    {
        var dispatcher = new CountingDispatcher();
        using var services = new TestServices(dispatcher: dispatcher);
        string path = Path.Combine(services.Directory, "model.slm");
        TestPackages.WriteSmall(path);
        services.Library.Entries.Add(new LibraryEntry { Id = "small", DisplayName = "small.slm", Path = path, Format = "SLM" });
        services.Precompiler.Run = (package, progress, token) =>
        {
            // A service that blocks the calling thread and reports every kernel.
            SlowWork.Block(StepWork, token);
            for (int kernel = 1; kernel <= 5000; kernel++)
            {
                progress.Report(new PrecompileProgress("vulkan", "GPU A", PrecompileStage.Portable, "Compiling", new LoadProgress(LoadStage.Compiling, kernel, 5000, "GPU A")));
            }

            SlowWork.Block(StepWork, token);
            return Task.FromResult(services.Precompiler.Report with { PackagePath = package });
        };
        var library = new LibraryViewModel(services.Services, new FakeNavigator());
        await library.RefreshAsync();
        var entry = library.Entries.Single();

        dispatcher.Reset();
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var heartbeat = UiHeartbeat.Start();
        await library.PrecompileAsync(entry);
        heartbeat.StopAndAssertResponsive("precompile");
        Assert.True(dispatcher.Posts <= PostBudget(watch.Elapsed), $"{dispatcher.Posts} UI jobs for 5000 kernel reports");
        Assert.Equal(1.0, entry.PrecompileRows.Single().Progress);

        // Converting: a slow service that logs a line per tensor.
        services.Library.Convert = (destination, log, token) =>
        {
            SlowWork.Block(StepWork, token);
            for (int tensor = 0; tensor < 5000; tensor++)
            {
                log.Report($"tensor {tensor}");
            }

            SlowWork.Block(StepWork, token);
            return destination;
        };
        var convert = new ConvertViewModel(services.Services, () => { }, () => { }) { Source = path, Destination = Path.Combine(services.Directory, "out.slm") };
        dispatcher.Reset();
        watch.Restart();
        heartbeat = UiHeartbeat.Start();
        await convert.ConvertCommand.ExecuteAsync(null);
        heartbeat.StopAndAssertResponsive("conversion");
        Assert.True(dispatcher.Posts <= PostBudget(watch.Elapsed), $"{dispatcher.Posts} UI jobs for 5000 log lines");
        Assert.Contains("tensor 4999", convert.Log, StringComparison.Ordinal);
        Assert.Contains("Done", convert.Log, StringComparison.Ordinal);
    });

    [Fact]
    public Task Downloads_ThrottleProgress() => HeadlessApp.RunAsync(async () =>
    {
        var dispatcher = new CountingDispatcher();
        using var services = new TestServices(dispatcher: dispatcher);
        services.Hub.ProgressFlood = 5000;
        var downloads = new DownloadsViewModel(services.Services);
        var heartbeat = UiHeartbeat.Start();
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var item = downloads.Enqueue(ModelReference.Parse("Qwen/Qwen3-0.6B-GGUF:Q8_0"), 1000);
        await services.Hub.Started.Task.WaitAsync(TestContext.Current.CancellationToken);
        await WaitForAsync(() => item.ReceivedBytes == 5400);
        heartbeat.StopAndAssertResponsive("download");
        Assert.True(dispatcher.Posts <= PostBudget(watch.Elapsed), $"{dispatcher.Posts} UI jobs for 5000 progress reports");
        services.Hub.Release();
        await WaitForAsync(() => item.Phase == PullPhase.Completed);
    });

    /// <summary>Runs the page's command once with the heartbeat on, and checks that the run completed.</summary>
    private static async Task RunResponsiveAsync(InferencePageViewModel page, System.Windows.Input.ICommand run, string what)
    {
        var heartbeat = UiHeartbeat.Start();
        Assert.True(run.CanExecute(null), $"{what}: the page cannot run");
        run.Execute(null);
        await page.Queue.Items[^1].Completion.WaitAsync(TestContext.Current.CancellationToken);
        heartbeat.StopAndAssertResponsive(what);

        // The final state is posted to the UI thread after the run's task completed.
        await WaitForAsync(() => !page.Queue.Items[^1].IsActive);
        Assert.Equal(RunStatus.Completed, page.Queue.Items[^1].Status);
    }

    /// <summary>Starts a long run, cancels it once it is running and checks that it ends cancelled soon after.</summary>
    private static async Task CancelMidRunAsync(InferencePageViewModel page, System.Windows.Input.ICommand run)
    {
        var heartbeat = UiHeartbeat.Start();
        run.Execute(null);
        var item = page.Queue.Items[^1];
        await WaitForAsync(() => item.Status == RunStatus.Running);
        await Task.Delay(StepWork / 2, TestContext.Current.CancellationToken);
        page.Queue.CancelAll();
        await item.Completion.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        heartbeat.StopAndAssertResponsive("cancel");
        await WaitForAsync(() => item.Status == RunStatus.Cancelled);
        Assert.Equal(RunStatus.Cancelled, item.Status);
    }

    private static NumberParameterViewModel Number(IEnumerable<ParameterViewModel> parameters, string key)
        => parameters.OfType<NumberParameterViewModel>().Single(parameter => parameter.Key == key);

    /// <summary>Awaits (keeping the UI thread free) until <paramref name="condition"/> holds, at most ten seconds.</summary>
    private static async Task WaitForAsync(Func<bool> condition)
    {
        for (int i = 0; i < 1000 && !condition(); i++)
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        Assert.True(condition(), "timed out");
    }
}
