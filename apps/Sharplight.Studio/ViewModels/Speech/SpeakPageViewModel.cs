using System.Diagnostics;
using System.Threading.Channels;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Audio;
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
        _ = RefreshSavedVoicesAsync(null);
        SaveVoiceCommand.NotifyCanExecuteChanged();
    }

    /// <inheritdoc/>
    /// <remarks>Lists the voices once the chosen model became resident (once per choice: listing leases the model).</remarks>
    protected override void OnModelStateChanged(ILoadedModel model)
    {
        if (model.State is ResidentState.Ready or ResidentState.Busy && !ReferenceEquals(_voicesReadFrom, model))
        {
            _voicesReadFrom = model;
            _ = RefreshVoicesAsync(model);
            _ = RefreshSavedVoicesAsync(null);
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

    /// <summary>The saved voices (the voice library in the storage layout's data folder).</summary>
    public VoiceLibrary VoiceLibrary => new(Services.StorageLocation.Current);

    /// <summary>The name a new saved voice gets.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SaveVoiceCommand))]
    public partial string VoiceName { get; set; } = "";

    /// <summary>A voice is being made from the reference recording.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SaveVoiceCommand))]
    public partial bool IsSavingVoice { get; set; }

    /// <summary>The microphone is recording the reference.</summary>
    [ObservableProperty]
    public partial bool IsRecordingReference { get; set; }

    /// <summary>The voice library has a voice for the chosen model.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(DeleteVoiceCommand))]
    public partial bool HasSavedVoices { get; set; }

    /// <summary>The text of the record button.</summary>
    public string RecordReferenceText => IsRecordingReference ? "Stop recording" : "Record a reference";

    /// <summary>Audio can be recorded (a microphone was found).</summary>
    public bool CanRecord => Services.Recorder.IsAvailable;

    partial void OnIsRecordingReferenceChanged(bool value) => OnPropertyChanged(nameof(RecordReferenceText));

    private IRecording? _referenceRecording;

    private string ReferenceRecordingPath => Path.Combine(Services.Pages.DirectoryFor("speak"), "recordings", "reference.wav");

    private ChoiceParameterViewModel SavedVoiceParameter => Panel.Parameters.OfType<ChoiceParameterViewModel>().First(candidate => candidate.Key == "savedvoice");

    private bool CanSaveVoice() => HasModel && !IsSavingVoice && VoiceName.Trim().Length > 0;

    /// <summary>Makes a reusable voice from the reference recording (and its words) with the chosen model, and saves it in the voice library.</summary>
    [RelayCommand(CanExecute = nameof(CanSaveVoice))]
    private async Task SaveVoiceAsync()
    {
        var model = SelectedModel!;
        string name = VoiceName.Trim();
        string path = Panel.GetText("reference").Trim();
        string referenceText = Panel.GetText("referencetext").Trim();
        if (path.Length == 0 || !File.Exists(path))
        {
            Notice = "Set the reference recording first (a WAV file, or record one).";
            return;
        }

        Notice = "";
        IsSavingVoice = true;
        try
        {
            var recording = await Task.Run(() => WavFile.Read(path));
            using var lease = await model.AcquireClientAsync<ISpeechSynthesizer>(CancellationToken.None);
            if (lease.Client.GetService(typeof(IVoiceCloner)) is not IVoiceCloner cloner)
            {
                Notice = "This model cannot clone a voice.";
                return;
            }

            var voice = await cloner.CreateVoiceAsync(name, recording, referenceText.Length > 0 ? referenceText : null);
            await Task.Run(() => VoiceLibrary.Save(voice));
            VoiceName = "";
            await RefreshSavedVoicesAsync(name);
        }
        catch (Exception error) when (error is IOException or InvalidOperationException or NotSupportedException or ArgumentException or InvalidDataException)
        {
            Notice = $"The voice could not be saved: {error.Message}";
        }
        finally
        {
            IsSavingVoice = false;
        }
    }

    private bool CanDeleteVoice() => HasSavedVoices;

    /// <summary>Deletes the chosen saved voice.</summary>
    [RelayCommand(CanExecute = nameof(CanDeleteVoice))]
    private async Task DeleteVoiceAsync()
    {
        string name = SavedVoiceParameter.Value;
        if (name.Length == 0)
        {
            return;
        }

        await Task.Run(() => VoiceLibrary.Delete(name));
        await RefreshSavedVoicesAsync(null);
    }

    /// <summary>Starts recording the microphone as the reference, or stops and takes the recording as the reference file.</summary>
    [RelayCommand]
    private async Task ToggleReferenceRecordingAsync()
    {
        if (_referenceRecording is { } running)
        {
            await running.StopAsync();
            running.Dispose();
            _referenceRecording = null;
            IsRecordingReference = false;
            Panel.Parameters.OfType<TextParameterViewModel>().First(candidate => candidate.Key == "reference").Deserialize(ReferenceRecordingPath);
            return;
        }

        if (!Services.Recorder.IsAvailable)
        {
            Notice = Services.Recorder.Description;
            return;
        }

        Notice = "";
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(ReferenceRecordingPath)!);
            _referenceRecording = Services.Recorder.Start(ReferenceRecordingPath);
            IsRecordingReference = true;
        }
        catch (Exception error) when (error is IOException or InvalidOperationException or System.ComponentModel.Win32Exception or NotSupportedException)
        {
            Notice = $"The microphone could not be recorded: {error.Message}";
        }
    }

    /// <summary>Lists the saved voices of the chosen model's family (all of them while no model is resident) and selects <paramref name="select"/> when given.</summary>
    /// <param name="select">The voice to choose, or <see langword="null"/> to keep the choice.</param>
    /// <returns>A task that completes when the choices are set.</returns>
    internal async Task RefreshSavedVoicesAsync(string? select)
    {
        string? architecture = null;
        if (SelectedModel is { State: ResidentState.Ready or ResidentState.Busy } model)
        {
            try
            {
                using var lease = await model.AcquireClientAsync<ISpeechSynthesizer>(CancellationToken.None);
                architecture = lease.Client is IVoiceCloner cloner ? cloner.VoiceArchitecture : lease.Client.GetService(typeof(IVoiceCloner)) is IVoiceCloner other ? other.VoiceArchitecture : "";
            }
            catch (Exception error) when (error is NotSupportedException or InvalidOperationException)
            {
                architecture = null;
            }
        }

        var names = architecture is { Length: 0 } ? [] : await Task.Run(() => VoiceLibrary.List(architecture));
        var parameter = SavedVoiceParameter;
        parameter.SetChoices([new ParameterChoice("", "None"), .. names.Select(name => new ParameterChoice(name, name))]);
        if (select is not null)
        {
            parameter.Deserialize(select);
        }

        HasSavedVoices = names.Count > 0;
        Panel.SetVisible("savedvoice", names.Count > 0);
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
        double? firstAudio = null;
        try
        {
            context.Report(null, "Loading the model");
            using var lease = await model.AcquireClientAsync<ISpeechSynthesizer>(context.CancellationToken).ConfigureAwait(false);
            context.Report(null, "Speaking");
            if (stream)
            {
                var samples = new List<float>();
                int segments = 0;
                var playback = Channel.CreateUnbounded<AudioBuffer>();
                playing = StartStreamPlayback(playback.Reader);
                try
                {
                    await foreach (var segment in lease.Client.SynthesizeStreamingAsync(text, options, context.CancellationToken).ConfigureAwait(false))
                    {
                        if (segments == 0)
                        {
                            firstAudio = watch.Elapsed.TotalSeconds;
                        }

                        samples.AddRange(segment.Samples);
                        segments++;
                        context.Report(null, $"{segments} chunks, {samples.Count / (double)lease.Client.SampleRate:0.0} s of audio");
                        playback.Writer.TryWrite(segment);
                    }
                }
                finally
                {
                    playback.Writer.TryComplete();
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
        if (firstAudio is { } first)
        {
            metadata["first audio"] = $"{first:0.00} s";
        }

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
        Services.Dispatcher.Post(() => IsPlaying = false);
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

    /// <summary>Plays the streamed chunks back to back on one output stream while they arrive, when the player exists.</summary>
    private Task StartStreamPlayback(ChannelReader<AudioBuffer> chunks)
    {
        if (!Services.Player.IsAvailable)
        {
            return Task.CompletedTask;
        }

        _playback ??= new CancellationTokenSource();
        var token = _playback.Token;
        Services.Dispatcher.Post(() => IsPlaying = true);
        return Task.Run(async () =>
        {
            try
            {
                await Services.Player.PlayStreamAsync(chunks.ReadAllAsync(token), token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Stopped by the user.
            }
            catch (Exception error) when (error is IOException or InvalidOperationException or NotSupportedException or System.ComponentModel.Win32Exception)
            {
                Services.Dispatcher.Post(() => Notice = $"The audio could not be played: {error.Message}");
            }
        }, CancellationToken.None);
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
            new ChoiceParameterViewModel("savedvoice", "Saved voice", "A voice made from a recording and saved in the voice library (cloning models); it replaces the voice and the reference recording.", [new("", "None")]),
            new NumberParameterViewModel("speed", "Speed", "Speaking rate, for models that support it.", 0.5, 2, 0.05, 1),
            new NumberParameterViewModel("temperature", "Temperature", "0 is greedy: the same text always sounds the same.", 0, 1.5, 0.05, 0.4),
            new NumberParameterViewModel("topk", "Top-k", "Sample from the k most likely audio tokens (0 = all).", 0, 100, 1, 40),
            new NumberParameterViewModel("topp", "Top-p", "Sample from the smallest set of tokens with this probability.", 0.1, 1, 0.05, 0.9),
            new NumberParameterViewModel("repetition", "Repetition penalty", "Above 1 discourages repeating audio tokens.", 1, 1.5, 0.05, 1.1),
            new TextParameterViewModel("seed", "Seed", "A whole number for repeatable output; empty is random.", "Random"),
            new TextParameterViewModel("reference", "Reference recording", "A WAV file of the voice to speak with (CosyVoice 3, F5-TTS, Chatterbox, XTTS, OmniVoice).", "Path to a .wav file"),
            new TextParameterViewModel("referencetext", "Reference text", "The words spoken in the reference recording (CosyVoice 3 zero-shot, F5-TTS; OmniVoice needs them).", "Optional"),
            new TextParameterViewModel("instructions", "Instructions", "How to speak, e.g. \"Please speak as fast as possible.\" (CosyVoice 3); the voice to design, e.g. \"female, british accent\" (OmniVoice).", "Optional"),
            new BoolParameterViewModel("stream", "Play while generating", "Speak chunk by chunk as the model generates them, for a short wait before the first sound (CosyVoice 3 and XTTS stream within a sentence, Chatterbox per clause)."),
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
            VoiceProfile = panel.GetChoice("savedvoice") is { Length: > 0 } saved && new VoiceLibrary(services.StorageLocation.Current).Contains(saved)
                ? new VoiceLibrary(services.StorageLocation.Current).Load(saved) : null,
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
