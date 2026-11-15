using Avalonia;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Renders the Transcribe and Speak pages with their shared blocks (picker, parameters, queue, history, outputs) and checks that the
/// views bind cleanly. Set <c>SHARPLIGHT_STUDIO_SCREENSHOTS</c> to a directory to also save each page as a PNG.
/// </summary>
public sealed class SpeechScreenshotTests
{
    private static readonly string? s_screenshots = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS");

    [Fact]
    public Task SpeechPages_RenderWithoutBindingErrors() => HeadlessApp.RunAsync(async () =>
    {
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        var recognizer = new FakeRecognizer();
        var whisper = new FakeModel("whisper-small") { Kind = ModelKinds.SpeechRecognition };
        whisper.Clients[typeof(ISpeechRecognizer)] = recognizer;
        var synthesizer = new FakeSynthesizer { Voices = ["female_1", "male_1"] };
        var outetts = new FakeModel("outetts-0.2") { Kind = ModelKinds.SpeechSynthesis };
        outetts.Clients[typeof(ISpeechSynthesizer)] = synthesizer;
        string wav = services.Directory + "/jfk.wav";
        WavFile.Write(wav, new AudioBuffer([.. Enumerable.Range(0, 16000 * 11).Select(i => 0.2f * MathF.Sin(2 * MathF.PI * 220 * i / 16000))], 16000));
        new ThemeService(Application.Current!).Apply(AppTheme.Dark, "Violet");
        var viewModel = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = viewModel, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            // Transcribe: no model, then a model with a finished run in the history and the gallery.
            await viewModel.NavigateAsync(viewModel.Transcribe);
            await HeadlessApp.RenderAsync(window);
            Assert.False(viewModel.Transcribe.HasModels);
            Save(window, "transcribe-empty");

            services.Runtime.Add(whisper);
            viewModel.OpenKind(ModelKinds.SpeechRecognition, whisper);
            await HeadlessApp.RenderAsync(window);
            await viewModel.Transcribe.SetAudioAsync(wav);
            viewModel.Transcribe.RunCommand.Execute(null);
            await viewModel.Transcribe.Queue.Items.Single().Completion;
            await HeadlessApp.RenderAsync(window);
            Assert.Equal(2, viewModel.Transcribe.Segments.Count);
            Save(window, "transcribe");

            viewModel.Transcribe.SideTab = 1;
            await HeadlessApp.RenderAsync(window);
            Save(window, "transcribe-history");
            viewModel.Transcribe.SideTab = 2;
            await HeadlessApp.RenderAsync(window);
            Save(window, "transcribe-outputs");
            viewModel.Transcribe.SideTab = 0;

            // Speak: with a model that offers voices and a finished run.
            services.Runtime.Add(outetts);
            viewModel.OpenKind(ModelKinds.SpeechSynthesis, outetts);
            await HeadlessApp.RenderAsync(window);
            viewModel.Speak.InputText = "Ask not what your country can do for you, ask what you can do for your country.";
            viewModel.Speak.RunCommand.Execute(null);
            await viewModel.Speak.Queue.Items.Single().Completion;
            await HeadlessApp.WaitForAsync(window, () => viewModel.Speak.HasResult);
            await HeadlessApp.RenderAsync(window);
            Assert.True(viewModel.Speak.HasResult);
            Save(window, "speak");

            // The Tasks page opens both pages.
            await viewModel.NavigateAsync(viewModel.Tasks);
            await HeadlessApp.RenderAsync(window);
            Assert.Contains(viewModel.Tasks.Kinds, card => card.Info.Id == ModelKinds.SpeechRecognition && card.IsRunnable);
            Save(window, "tasks-runnable");
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    private static void Save(MainWindow window, string name)
    {
        using var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        if (s_screenshots is null)
        {
            return;
        }

        Directory.CreateDirectory(s_screenshots);
        frame.Save(Path.Combine(s_screenshots, $"studio-{name}.png"), new PngBitmapEncoderOptions());
    }
}
