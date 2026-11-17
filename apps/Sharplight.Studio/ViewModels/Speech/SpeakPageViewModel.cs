using System.Diagnostics;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.ViewModels.Speech;

/// <summary>
/// The Speak page (kind <c>speech-synthesis</c>): text goes to the chosen model's <see cref="ISpeechSynthesizer"/> and the audio can
/// be played (through the system's player program), saved as WAV or opened in the system's audio program. With <c>Play while
/// generating</c> the segments the model streams are played as they arrive. Built from the shared inference blocks.
/// </summary>
internal sealed partial class SpeakPageViewModel : InferencePageViewModel
{
    private const string AudioName = "speech.wav";
    private CancellationTokenSource? _playback;
    private string? _resultPath;

    /// <summary>The chosen model whose voices were listed, or null.</summary>
    private ILoadedModel? _voicesReadFrom;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    public SpeakPageViewModel(StudioServices services, INavigator navigator)
        : base(services, navigator, ModelKinds.SpeechSynthesis, "speak", CreateParameters(services))
    {
    }

    /// <inheritdoc/>
    public override string Title => "Speak";

    /// <inheritdoc/>
    public override string Icon => "IconSpeaker";

    /// <inheritdoc/>
    public override string KindNoun => "text to speech";

    /// <summary>The typed parameter panel (maps to <see cref="SpeechSynthesisOptions"/>).</summary>
    public ParameterPanelViewModel<SpeechSynthesisOptions> Panel => (ParameterPanelViewModel<SpeechSynthesisOptions>)Parameters;

    /// <summary>The text to speak.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RunCommand))]
    [NotifyPropertyChangedFor(nameof(RunHint))]
    public partial string InputText { get; set; } = "";

    /// <summary>The page is synthesizing (a run is queued or running).</summary>
    public bool IsSynthesizing => Queue.IsBusy;

    /// <summary>The audio is playing.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(PlayText))]
    public partial bool IsPlaying { get; set; }

    /// <summary>Length, rate and speed of the result, or empty while there is none.</summary>
    [ObservableProperty]
    public partial string ResultSummary { get; set; } = "";

    /// <summary>A result is ready to play and export.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand), nameof(SaveCommand), nameof(OpenResultCommand))]
    public partial bool HasResult { get; set; }

    /// <summary>The text of the play button.</summary>
    public string PlayText => IsPlaying ? "Stop" : "Play";

    /// <summary>Audio can be played inside the app (a player program was found).</summary>
    public bool CanPlay => Services.Player.IsAvailable;

    /// <summary>What plays the sound, or why nothing can.</summary>
    public string PlayHint => Services.Player.Description;

    /// <summary>Why <see cref="RunCommand"/> cannot run, for the hint under the button; empty when it can.</summary>
    public string RunHint => !HasModel ? "Load a text-to-speech model first." : InputText.Trim().Length == 0 ? "Type the text to speak." : "";

    /// <inheritdoc/>
    protected override void OnModelChanged(ILoadedModel? model)
    {
        OnPropertyChanged(nameof(RunHint));
        RunCommand.NotifyCanExecuteChanged();
        _voicesReadFrom = model is { State: ResidentState.Ready or ResidentState.Busy } ? model : null;
        _ = RefreshVoicesAsync(model);
    }

    /// <inheritdoc/>
    /// <remarks>Lists the voices once the chosen model became resident (once per choice: listing leases the model).</remarks>
    protected override void OnModelStateChanged(ILoadedModel model)
    {
        if (model.State is ResidentState.Ready or ResidentState.Busy && !ReferenceEquals(_voicesReadFrom, model))
        {
            _voicesReadFrom = model;
            _ = RefreshVoicesAsync(model);
        }
    }

    /// <inheritdoc/>
    protected override void OnBusyChanged()
    {
        OnPropertyChanged(nameof(IsSynthesizing));
        RunCommand.NotifyCanExecuteChanged();
    }

    /// <inheritdoc/>
    protected override void Restore(RunRecord record)
    {
        string path = Archive.PathOf(record, AudioName);
        Panel.Apply(record.Options);
        if (record.Input.TryGetValue("text", out var text))
        {
            InputText = text;
        }

        if (File.Exists(path))
        {
            ShowResult(path, string.Join(" · ", record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}")));
        }
        else
        {
            Notice = "The stored audio is gone.";
        }
    }

    /// <summary>Lists the voices of the model when it is resident (asking a sleeping model would load it), and hides the voice choice when it has none.</summary>
    /// <param name="model">The chosen model.</param>
    /// <returns>A task that completes when the voices are listed.</returns>
    internal async Task RefreshVoicesAsync(ILoadedModel? model)
    {
        IReadOnlyList<string> voices = [];
        if (model is { State: ResidentState.Ready or ResidentState.Busy })
        {
            try
            {
                using var lease = await model.AcquireClientAsync<ISpeechSynthesizer>(CancellationToken.None);
                voices = lease.Client.Voices;
            }
            catch (Exception error) when (error is NotSupportedException or InvalidOperationException)
            {
                voices = [];
            }
        }

        var parameter = Panel.Parameters.OfType<ChoiceParameterViewModel>().First(candidate => candidate.Key == "voice");
        parameter.SetChoices([new ParameterChoice("", voices.Count == 0 ? "Model's own voice" : "Default voice"), .. voices.Select(voice => new ParameterChoice(voice, voice))]);
        Panel.SetVisible("voice", voices.Count > 0);
    }

    private bool CanRun() => HasModel && InputText.Trim().Length > 0;

    /// <summary>Speaks the text with the chosen model and the current parameters.</summary>
    [RelayCommand(CanExecute = nameof(CanRun))]
    private void Run()
    {
        var model = SelectedModel!;
        string text = InputText.Trim();
        var options = Panel.Options;
        bool stream = Panel.GetBool("stream");
        var values = Panel.Snapshot();
        Panel.Persist();
        Notice = "";
        StopPlayback();
        HasResult = false;
        ResultSummary = "";
        Queue.Enqueue(text.Length > 60 ? text[..60] + "…" : text, model.Id, context => SynthesizeAsync(model, text, options, stream, values, context));
        OnPropertyChanged(nameof(IsSynthesizing));
        RunCommand.NotifyCanExecuteChanged();
    }

    private async Task SynthesizeAsync(ILoadedModel model, string text, SpeechSynthesisOptions options, bool stream, Dictionary<string, string> values, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        AudioBuffer? audio = null;
        string? error = null;
        RunOutcome outcome = RunOutcome.Completed;
        string title = text.Length > 60 ? text[..60] + "…" : text;
        Task playing = Task.CompletedTask;
        try
        {
            context.Report(null, "Loading the model");
            using var lease = await model.AcquireClientAsync<ISpeechSynthesizer>(context.CancellationToken).ConfigureAwait(false);
            context.Report(null, "Speaking");
            if (stream)
            {
                var samples = new List<float>();
                int segments = 0;
                await foreach (var segment in lease.Client.SynthesizeStreamingAsync(text, options, context.CancellationToken).ConfigureAwait(false))
                {
                    samples.AddRange(segment.Samples);
                    segments++;
                    context.Report(null, $"{segments} segments, {samples.Count / (double)lease.Client.SampleRate:0.0} s of audio");
                    playing = PlaySegment(playing, segment, segments);
                }

                audio = new AudioBuffer([.. samples], lease.Client.SampleRate);
            }
            else
            {
                audio = (await lease.Client.SynthesizeAsync(text, options, context.CancellationToken).ConfigureAwait(false)).Audio;
            }
        }
        catch (OperationCanceledException)
        {
            outcome = RunOutcome.Cancelled;
        }
        catch (Exception failure) when (failure is not OutOfMemoryException)
        {
            outcome = RunOutcome.Failed;
            error = failure.GetBaseException().Message;
        }

        double seconds = watch.Elapsed.TotalSeconds;
        var metadata = new Dictionary<string, string> { ["time"] = $"{seconds:0.0} s" };
        Dictionary<string, byte[]>? files = null;
        if (outcome == RunOutcome.Completed && audio is not null)
        {
            using var stored = new MemoryStream();
            WavFile.Write(stored, audio);
            files = new Dictionary<string, byte[]> { [AudioName] = stored.ToArray() };
            metadata["audio"] = AudioFormat.Length(audio.Duration);
            metadata["rate"] = $"{audio.SampleRate / 1000.0:0.#} kHz";
            metadata["speed"] = $"{seconds / Math.Max(0.001, audio.Duration.TotalSeconds):0.00}x real-time factor";
            if (!string.IsNullOrEmpty(options.Voice))
            {
                metadata["voice"] = options.Voice;
            }
        }

        var record = new RunRecord
        {
            Id = PageArchive.NewId(),
            Title = title,
            Model = model.Id,
            Outcome = outcome,
            Seconds = seconds,
            Error = error,
            Input = new Dictionary<string, string> { ["text"] = text },
            Options = values,
            Metadata = metadata,
            Preview = text.Length > 240 ? text[..240] + "…" : text,
            Files = files is null ? [] : [AudioName],
        };
        if (outcome != RunOutcome.Cancelled)
        {
            Archive.WriteFiles(record, files);
            Services.Dispatcher.Post(() =>
            {
                Archive.Add(record);
                if (files is not null)
                {
                    ShowResult(Archive.PathOf(record, AudioName), string.Join(" · ", metadata.Select(pair => $"{pair.Key}: {pair.Value}")));
                }
            });
        }

        await playing.ConfigureAwait(false);
        if (outcome == RunOutcome.Cancelled)
        {
            throw new OperationCanceledException(context.CancellationToken);
        }

        if (error is not null)
        {
            Services.Dispatcher.Post(() => Notice = error);
            throw new InvalidOperationException(error);
        }
    }

    /// <summary>Plays a streamed segment after the ones before it, when the player exists.</summary>
    private Task PlaySegment(Task previous, AudioBuffer segment, int index)
    {
        if (!Services.Player.IsAvailable)
        {
            return previous;
        }

        string path = Path.Combine(Services.Pages.DirectoryFor("speak"), "stream", $"segment-{index}.wav");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        WavFile.Write(path, segment);
        _playback ??= new CancellationTokenSource();
        var token = _playback.Token;
        Services.Dispatcher.Post(() => IsPlaying = true);
        return previous.ContinueWith(async _ =>
        {
            try
            {
                await Services.Player.PlayAsync(path, token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Stopped by the user.
            }
        }, CancellationToken.None, TaskContinuationOptions.None, TaskScheduler.Default).Unwrap();
    }

    private void ShowResult(string path, string summary)
    {
        _resultPath = path;
        ResultSummary = summary;
        HasResult = true;
        IsPlaying = false;
    }

    private bool CanUseResult() => HasResult;

    /// <summary>Plays the result in the system's audio output, or stops it.</summary>
    [RelayCommand(CanExecute = nameof(CanUseResult))]
    private async Task PlayAsync()
    {
        if (IsPlaying)
        {
            StopPlayback();
            return;
        }

        if (_resultPath is not { } path)
        {
            return;
        }

        if (!Services.Player.IsAvailable)
        {
            Notice = Services.Player.Description;
            Services.Platform.Open(path);
            return;
        }

        _playback = new CancellationTokenSource();
        var token = _playback.Token;
        IsPlaying = true;
        try
        {
            await Services.Player.PlayAsync(path, token);
        }
        catch (OperationCanceledException)
        {
            // Stopped by the user.
        }
        catch (Exception error) when (error is IOException or InvalidOperationException or System.ComponentModel.Win32Exception)
        {
            Notice = $"The audio could not be played: {error.Message}";
        }
        finally
        {
            IsPlaying = false;
        }
    }

    private void StopPlayback()
    {
        _playback?.Cancel();
        _playback = null;
        IsPlaying = false;
    }

    /// <summary>Saves the result as a WAV file where the user chooses.</summary>
    [RelayCommand(CanExecute = nameof(CanUseResult))]
    private async Task SaveAsync()
    {
        if (_resultPath is { } path && File.Exists(path))
        {
            await Exporter.SaveAsync(Services.Platform, "Save audio", "speech.wav", await File.ReadAllBytesAsync(path));
        }
    }

    /// <summary>Opens the result in the system's audio program.</summary>
    [RelayCommand(CanExecute = nameof(CanUseResult))]
    private void OpenResult()
    {
        if (_resultPath is { } path)
        {
            Services.Platform.Open(path);
        }
    }

    /// <summary>Cancels the queued and running syntheses and stops playback.</summary>
    [RelayCommand]
    private void Cancel()
    {
        Queue.CancelAll();
        StopPlayback();
    }

    private static ParameterPanelViewModel<SpeechSynthesisOptions> CreateParameters(StudioServices services)
    {
        ParameterViewModel[] parameters =
        [
            new ChoiceParameterViewModel("voice", "Voice", "The speakers the model offers.", [new("", "Model's own voice")]),
            new NumberParameterViewModel("speed", "Speed", "Speaking rate, for models that support it.", 0.5, 2, 0.05, 1),
            new NumberParameterViewModel("temperature", "Temperature", "0 is greedy: the same text always sounds the same.", 0, 1.5, 0.05, 0.4),
            new NumberParameterViewModel("topk", "Top-k", "Sample from the k most likely audio tokens (0 = all).", 0, 100, 1, 40),
            new NumberParameterViewModel("topp", "Top-p", "Sample from the smallest set of tokens with this probability.", 0.1, 1, 0.05, 0.9),
            new NumberParameterViewModel("repetition", "Repetition penalty", "Above 1 discourages repeating audio tokens.", 1, 1.5, 0.05, 1.1),
            new TextParameterViewModel("seed", "Seed", "A whole number for repeatable output; empty is random.", "Random"),
            new TextParameterViewModel("reference", "Reference recording", "A WAV file of the voice to speak with (CosyVoice 3, F5-TTS, Chatterbox, XTTS).", "Path to a .wav file"),
            new TextParameterViewModel("referencetext", "Reference text", "The words spoken in the reference recording (CosyVoice 3 zero-shot, F5-TTS).", "Optional"),
            new TextParameterViewModel("instructions", "Instructions", "How to speak, e.g. \"Please speak as fast as possible.\" (CosyVoice 3).", "Optional"),
            new BoolParameterViewModel("stream", "Play while generating", "Speak segment by segment as the model streams them (sounds slightly different from a whole-text pass)."),
        ];
        ParameterPreset[] presets =
        [
            new("Natural", new() { ["temperature"] = "0.4", ["topk"] = "40", ["topp"] = "0.9", ["repetition"] = "1.1" }, BuiltIn: true),
            new("Deterministic", new() { ["temperature"] = "0", ["topk"] = "1", ["topp"] = "1", ["repetition"] = "1" }, BuiltIn: true),
            new("Expressive", new() { ["temperature"] = "0.8", ["topk"] = "60", ["topp"] = "0.95", ["repetition"] = "1.1" }, BuiltIn: true),
        ];
        return new ParameterPanelViewModel<SpeechSynthesisOptions>(services.Pages, "speak", parameters, panel => new SpeechSynthesisOptions
        {
            Voice = panel.GetChoice("voice") is { Length: > 0 } voice ? voice : null,
            Speed = (float)panel.GetNumber("speed"),
            ReferenceAudio = panel.GetText("reference").Trim() is { Length: > 0 } reference && File.Exists(reference) ? WavFile.Read(reference) : null,
            ReferenceText = panel.GetText("referencetext").Trim() is { Length: > 0 } referenceText ? referenceText : null,
            Instructions = panel.GetText("instructions").Trim() is { Length: > 0 } instructions ? instructions : null,
            Sampling = new SamplingOptions
            {
                Temperature = (float)panel.GetNumber("temperature"),
                TopK = panel.GetInt("topk"),
                TopP = (float)panel.GetNumber("topp"),
                RepetitionPenalty = (float)panel.GetNumber("repetition"),
                Seed = int.TryParse(panel.GetText("seed").Trim(), out int seed) ? seed : null,
            },
        }, presets);
    }
}
