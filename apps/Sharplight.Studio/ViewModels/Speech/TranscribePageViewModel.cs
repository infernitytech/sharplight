using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Audio;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.ViewModels.Speech;

/// <summary>
/// The Transcribe page (kind <c>speech-recognition</c>): audio from a WAV file or the microphone goes to the chosen model's
/// <see cref="ISpeechRecognizer"/>; the timestamped segments appear while the model decodes, and the transcript can be copied or
/// exported as text, SubRip or WebVTT. Built from the shared inference blocks (picker, parameters, queue, history, outputs).
/// </summary>
internal sealed partial class TranscribePageViewModel : InferencePageViewModel
{
    private const string TranscriptName = "transcript.json";
    private static readonly FileFilter s_audioFiles = new("WAV audio", ["*.wav"]);
    private static readonly FileFilter s_allFiles = new("All files", ["*"]);

    private static readonly ParameterChoice[] s_languages =
    [
        new("", "Detect the language"), new("en", "English"), new("de", "German"), new("fr", "French"), new("es", "Spanish"),
        new("it", "Italian"), new("pt", "Portuguese"), new("nl", "Dutch"), new("pl", "Polish"), new("sv", "Swedish"),
        new("da", "Danish"), new("no", "Norwegian"), new("fi", "Finnish"), new("cs", "Czech"), new("ru", "Russian"),
        new("uk", "Ukrainian"), new("tr", "Turkish"), new("el", "Greek"), new("ar", "Arabic"), new("he", "Hebrew"),
        new("hi", "Hindi"), new("zh", "Chinese"), new("ja", "Japanese"), new("ko", "Korean"), new("vi", "Vietnamese"),
        new("th", "Thai"), new("id", "Indonesian"),
    ];

    private AudioBuffer? _audio;
    private IRecording? _recording;
    private TranscriptFile? _result;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    public TranscribePageViewModel(StudioServices services, INavigator navigator)
        : base(services, navigator, ModelKinds.SpeechRecognition, "transcribe", CreateParameters(services))
    {
    }

    /// <inheritdoc/>
    public override string Title => "Transcribe";

    /// <inheritdoc/>
    public override string Icon => "IconMic";

    /// <inheritdoc/>
    public override string KindNoun => "speech to text";

    /// <summary>The typed parameter panel (maps to <see cref="SpeechRecognitionOptions"/>).</summary>
    public ParameterPanelViewModel<SpeechRecognitionOptions> Panel => (ParameterPanelViewModel<SpeechRecognitionOptions>)Parameters;

    /// <summary>The audio file shown as the input (a recording, or the file the user chose).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(AudioName))]
    public partial string AudioPath { get; set; } = "";

    /// <summary>The name of the input file.</summary>
    public string AudioName => Path.GetFileName(AudioPath);

    /// <summary>Length, rate and channels of the input, or why it cannot be read.</summary>
    [ObservableProperty]
    public partial string AudioSummary { get; set; } = "";

    /// <summary>The microphone is recording.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(RecordText))]
    public partial bool IsRecording { get; set; }

    /// <summary>The page is transcribing (a run is queued or running).</summary>
    public bool IsTranscribing => Queue.IsBusy;

    /// <summary>The segments of the result, in order.</summary>
    public ObservableCollection<TranscriptSegmentViewModel> Segments { get; } = [];

    /// <summary>The result's whole text.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasResult))]
    public partial string TranscriptText { get; set; } = "";

    /// <summary>The language of the result, or empty.</summary>
    [ObservableProperty]
    public partial string ResultLanguage { get; set; } = "";

    /// <summary>Time, audio length and speed of the last run.</summary>
    [ObservableProperty]
    public partial string ResultSummary { get; set; } = "";

    /// <summary>There is a result to show.</summary>
    public bool HasResult => TranscriptText.Length > 0 || Segments.Count > 0;

    /// <summary>Audio is chosen or recorded.</summary>
    public bool HasAudio => _audio is not null;

    /// <summary>The microphone can be recorded on this machine.</summary>
    public bool CanRecord => Services.Recorder.IsAvailable;

    /// <summary>What records the microphone, or why it cannot.</summary>
    public string RecordHint => Services.Recorder.Description;

    /// <summary>The text of the record button.</summary>
    public string RecordText => IsRecording ? "Stop recording" : "Record";

    /// <summary>Why <see cref="RunCommand"/> cannot run, for the hint under the button; empty when it can.</summary>
    public string RunHint => !HasModel ? "Load a speech-to-text model first." : !HasAudio ? "Choose or record audio first." : "";

    /// <summary>
    /// Takes <paramref name="path"/> as the input: reads the WAV file and shows its length. Other formats are refused with a notice,
    /// because the engine's audio I/O reads WAV only.
    /// </summary>
    /// <param name="path">The audio file.</param>
    /// <returns>Whether the file is usable.</returns>
    public async Task<bool> SetAudioAsync(string path)
    {
        Notice = "";
        if (!File.Exists(path))
        {
            Notice = $"{path} does not exist.";
            return false;
        }

        if (!path.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
        {
            Notice = $"{Path.GetFileName(path)} is not a WAV file. Studio reads WAV audio only; convert it first, for example with ffmpeg -i input.mp3 output.wav.";
            return false;
        }

        try
        {
            var audio = await Task.Run(() => WavFile.Read(path));
            _audio = audio;
            AudioPath = path;
            AudioSummary = $"{AudioFormat.Length(audio.Duration)} · {audio.SampleRate / 1000.0:0.#} kHz · {(audio.Channels == 1 ? "mono" : audio.Channels + " channels")}";
            OnInputChanged();
            return true;
        }
        catch (Exception error) when (error is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException)
        {
            Notice = $"{Path.GetFileName(path)} cannot be read: {error.Message}";
            return false;
        }
    }

    /// <inheritdoc/>
    protected override void OnModelChanged(ILoadedModel? model) => OnInputChanged();

    /// <inheritdoc/>
    protected override void OnBusyChanged()
    {
        OnPropertyChanged(nameof(IsTranscribing));
        RunCommand.NotifyCanExecuteChanged();
    }

    /// <inheritdoc/>
    protected override void Restore(RunRecord record)
    {
        string path = Archive.PathOf(record, TranscriptName);
        if (!File.Exists(path))
        {
            Notice = "The stored transcript is gone.";
            return;
        }

        var file = PageJson.Read(path, StudioJsonContext.Default.TranscriptFile);
        Panel.Apply(record.Options);
        Show(file);
        ResultSummary = string.Join(" · ", record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}"));
        if (record.Input.TryGetValue("file", out var name))
        {
            AudioPath = name;
        }
    }

    private void OnInputChanged()
    {
        OnPropertyChanged(nameof(HasAudio));
        OnPropertyChanged(nameof(RunHint));
        RunCommand.NotifyCanExecuteChanged();
    }

    private bool CanRun() => HasModel && HasAudio && !IsRecording;

    /// <summary>Opens the file picker for the input audio.</summary>
    [RelayCommand]
    private async Task ChooseFileAsync()
    {
        var files = await Services.Platform.PickFilesAsync("Choose audio", [s_audioFiles, s_allFiles], allowMultiple: false);
        if (files.Count > 0)
        {
            await SetAudioAsync(files[0]);
        }
    }

    /// <summary>Starts recording the microphone, or stops and takes the recording as the input.</summary>
    [RelayCommand]
    private async Task ToggleRecordingAsync()
    {
        if (_recording is { } running)
        {
            await running.StopAsync();
            running.Dispose();
            _recording = null;
            IsRecording = false;
            RunCommand.NotifyCanExecuteChanged();
            await SetAudioAsync(RecordingPath);
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
            Directory.CreateDirectory(Path.GetDirectoryName(RecordingPath)!);
            _recording = Services.Recorder.Start(RecordingPath);
            IsRecording = true;
            RunCommand.NotifyCanExecuteChanged();
        }
        catch (Exception error) when (error is IOException or InvalidOperationException or System.ComponentModel.Win32Exception or NotSupportedException)
        {
            Notice = $"The microphone could not be recorded: {error.Message}";
        }
    }

    private string RecordingPath => Path.Combine(Services.Pages.DirectoryFor("transcribe"), "recordings", "recording.wav");

    /// <summary>Transcribes the input with the chosen model and the current parameters.</summary>
    [RelayCommand(CanExecute = nameof(CanRun))]
    private void Run()
    {
        var model = SelectedModel!;
        var audio = _audio!;
        var options = Panel.Options;
        var values = Panel.Snapshot();
        string name = Path.GetFileName(AudioPath);
        Panel.Persist();
        Notice = "";
        Segments.Clear();
        TranscriptText = "";
        ResultLanguage = "";
        ResultSummary = "";
        _result = null;
        Queue.Enqueue(name, model.Id, context => TranscribeAsync(model, audio, options, values, name, context));
        OnPropertyChanged(nameof(IsTranscribing));
        RunCommand.NotifyCanExecuteChanged();
    }

    private async Task TranscribeAsync(ILoadedModel model, AudioBuffer audio, SpeechRecognitionOptions options, Dictionary<string, string> values, string name, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        var segments = new List<TranscriptSegmentRecord>();
        string? language = options.Language;
        RunOutcome outcome = RunOutcome.Completed;
        string? error = null;
        try
        {
            context.Report(null, "Loading the model");
            using var lease = await model.AcquireClientAsync<ISpeechRecognizer>(context.CancellationToken).ConfigureAwait(false);
            context.Report(0, "Transcribing");
            double total = Math.Max(0.001, audio.Duration.TotalSeconds);
            await foreach (var segment in lease.Client.TranscribeStreamingAsync(audio, options, context.CancellationToken).ConfigureAwait(false))
            {
                var record = new TranscriptSegmentRecord(segment.Start.TotalSeconds, segment.End.TotalSeconds, segment.Text);
                segments.Add(record);
                language ??= segment.Language;
                Services.Dispatcher.Post(() => Segments.Add(new TranscriptSegmentViewModel(record)));
                context.Report(Math.Min(1, record.End / total), $"{TranscriptFile.Clock(record.End, '.')[3..8]} of {AudioFormat.Length(audio.Duration)}");
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
        var file = new TranscriptFile { Segments = segments, Language = language, DurationSeconds = audio.Duration.TotalSeconds };
        file.Text = file.ToText();
        var metadata = new Dictionary<string, string>
        {
            ["audio"] = AudioFormat.Length(audio.Duration),
            ["time"] = $"{seconds:0.0} s",
            ["speed"] = $"{audio.Duration.TotalSeconds / Math.Max(0.001, seconds):0.#}x real time",
        };
        if (language is not null)
        {
            metadata["language"] = language;
        }

        if (outcome == RunOutcome.Completed)
        {
            Services.Dispatcher.Post(() =>
            {
                Show(file);
                ResultSummary = string.Join(" · ", metadata.Select(pair => $"{pair.Key}: {pair.Value}"));
            });
        }

        var record2 = new RunRecord
        {
            Id = PageArchive.NewId(),
            Title = name,
            Model = model.Id,
            Outcome = outcome,
            Seconds = seconds,
            Error = error,
            Input = new Dictionary<string, string> { ["file"] = name },
            Options = values,
            Metadata = metadata,
            Preview = file.Text.Length > 240 ? file.Text[..240] + "…" : file.Text,
            Files = outcome == RunOutcome.Completed ? ["transcript.txt", "transcript.srt", "transcript.vtt", TranscriptName] : [],
        };
        var files = outcome != RunOutcome.Completed ? null : new Dictionary<string, byte[]>
        {
            ["transcript.txt"] = System.Text.Encoding.UTF8.GetBytes(file.ToText() + "\n"),
            ["transcript.srt"] = System.Text.Encoding.UTF8.GetBytes(file.ToSrt()),
            ["transcript.vtt"] = System.Text.Encoding.UTF8.GetBytes(file.ToVtt()),
            [TranscriptName] = JsonSerializer.SerializeToUtf8Bytes(file, StudioJsonContext.Default.TranscriptFile),
        };
        if (outcome != RunOutcome.Cancelled)
        {
            Archived(record2, files);
        }

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

    private void Show(TranscriptFile file)
    {
        _result = file;
        Segments.Clear();
        foreach (var segment in file.Segments)
        {
            Segments.Add(new TranscriptSegmentViewModel(segment));
        }

        TranscriptText = file.ToText();
        ResultLanguage = file.Language ?? "";
        OnPropertyChanged(nameof(HasResult));
    }

    /// <summary>Copies the transcript's text to the clipboard.</summary>
    [RelayCommand]
    private Task CopyAsync() => Services.Platform.CopyTextAsync(TranscriptText);

    /// <summary>Saves the transcript as <c>txt</c>, <c>srt</c> or <c>vtt</c> where the user chooses.</summary>
    /// <param name="format">The extension.</param>
    [RelayCommand]
    private async Task ExportAsync(string format)
    {
        if (_result is not { } file)
        {
            return;
        }

        string text = format switch
        {
            "srt" => file.ToSrt(),
            "vtt" => file.ToVtt(),
            _ => file.ToText() + "\n",
        };
        string stem = Path.GetFileNameWithoutExtension(AudioPath);
        await Exporter.SaveTextAsync(Services.Platform, "Export transcript", $"{(stem.Length == 0 ? "transcript" : stem)}.{format}", text);
    }

    /// <summary>Cancels the queued and running transcriptions.</summary>
    [RelayCommand]
    private void Cancel() => Queue.CancelAll();

    private static ParameterPanelViewModel<SpeechRecognitionOptions> CreateParameters(StudioServices services)
    {
        ParameterViewModel[] parameters =
        [
            new ChoiceParameterViewModel("language", "Language", "Detected from the audio when not set.", s_languages),
            new ChoiceParameterViewModel("task", "Task", "Translate writes English whatever the language spoken.",
                [new("transcribe", "Transcribe"), new("translate", "Translate to English")]),
            new ChoiceParameterViewModel("timestamps", "Timestamps", "Segments carry start and end times; without them the model decodes one segment per 30 s window.",
                [new("segment", "Segments"), new("none", "None")]),
            new NumberParameterViewModel("beam", "Beam size", "1 is greedy decoding; more beams are slower and usually more accurate.", 1, 8, 1, 1),
            new BoolParameterViewModel("condition", "Condition on previous text", "Prompt each 30 s window with the text so far (long recordings)."),
            new TextParameterViewModel("prompt", "Prompt", "Names and terms that should be spelled right.", "Optional initial prompt"),
        ];
        ParameterPreset[] presets =
        [
            new("Fast", new() { ["beam"] = "1", ["timestamps"] = "segment" }, BuiltIn: true),
            new("Accurate", new() { ["beam"] = "5", ["timestamps"] = "segment" }, BuiltIn: true),
            new("Long recording", new() { ["beam"] = "1", ["condition"] = "true", ["timestamps"] = "segment" }, BuiltIn: true),
        ];
        return new ParameterPanelViewModel<SpeechRecognitionOptions>(services.Pages, "transcribe", parameters, panel => new SpeechRecognitionOptions
        {
            Language = panel.GetChoice("language") is { Length: > 0 } language ? language : null,
            Translate = panel.GetChoice("task") == "translate",
            Timestamps = panel.GetChoice("timestamps") == "none" ? TimestampGranularity.None : TimestampGranularity.Segment,
            BeamSize = panel.GetInt("beam"),
            ConditionOnPreviousText = panel.GetBool("condition"),
            Prompt = panel.GetText("prompt") is { Length: > 0 } prompt ? prompt : null,
        }, presets);
    }
}
