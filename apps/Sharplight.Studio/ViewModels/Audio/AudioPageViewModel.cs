using System.Collections.ObjectModel;
using System.Diagnostics;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;

namespace Sharplight.Studio.ViewModels.Audio;

/// <summary>
/// The Audio page, for the audio kinds that are not speech: an audio codec (<c>audio-codec</c>: the input is encoded to codes and decoded again, the
/// reconstruction can be played and exported) and a voice activity detector (<c>voice-activity</c>: the speech segments of the input on a timeline).
/// The mode follows the kind of the chosen model. Built from the shared inference blocks; playback is the Speak page's (the system's player program).
/// </summary>
internal sealed partial class AudioPageViewModel : InferencePageViewModel
{
    private const string AudioName = "reconstruction.wav";
    private const string SegmentsName = "segments.txt";
    private static readonly FileFilter s_audioFiles = new("Audio files", ["*.wav", "*.flac", "*.mp3", "*.ogg", "*.oga", "*.opus"]);
    private static readonly FileFilter s_allFiles = new("All files", ["*"]);

    private AudioBuffer? _audio;
    private CancellationTokenSource? _playback;
    private string? _resultPath;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    public AudioPageViewModel(StudioServices services, INavigator navigator)
        : base(services, navigator, [ModelKinds.AudioCodec, ModelKinds.VoiceActivity], "audio", CreateParameters(services))
    {
        ApplyMode();
    }

    /// <inheritdoc/>
    public override string Title => "Audio";

    /// <inheritdoc/>
    public override string Icon => "IconWave";

    /// <inheritdoc/>
    public override string KindNoun => "audio";

    /// <summary>The typed parameter panel.</summary>
    public ParameterPanelViewModel<AudioSettings> Panel => (ParameterPanelViewModel<AudioSettings>)Parameters;

    /// <summary>What the chosen model does (from its kind).</summary>
    public AudioMode Mode { get; private set; }

    /// <summary>The audio file shown as the input.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(AudioName2))]
    public partial string AudioPath { get; set; } = "";

    /// <summary>The name of the input file.</summary>
    public string AudioName2 => Path.GetFileName(AudioPath);

    /// <summary>Length, rate and channels of the input, or why it cannot be read.</summary>
    [ObservableProperty]
    public partial string AudioSummary { get; set; } = "";

    /// <summary>The page is running (a run is queued or running).</summary>
    public bool IsRunning => Queue.IsBusy;

    /// <summary>Audio is chosen.</summary>
    public bool HasAudio => _audio is not null;

    /// <summary>The speech segments of the last voice activity run.</summary>
    public ObservableCollection<SpeechSegmentViewModel> Segments { get; } = [];

    /// <summary>The page shows the codec result (the reconstruction).</summary>
    public bool IsCodec => Mode == AudioMode.Codec;

    /// <summary>The page shows the voice activity result (the segments).</summary>
    public bool IsVoiceActivity => Mode == AudioMode.VoiceActivity;

    /// <summary>The text of the run button.</summary>
    public string RunText => Mode switch { AudioMode.Codec => "Round trip", AudioMode.VoiceActivity => "Find speech", _ => "Run" };

    /// <summary>What the chosen mode does, for the hint above the input.</summary>
    public string ModeDescription => Mode switch
    {
        AudioMode.Codec => "The codec encodes the audio to discrete codes and decodes them again; play the reconstruction and compare.",
        AudioMode.VoiceActivity => "The detector marks the stretches of speech on a timeline.",
        _ => "Load an audio codec or a voice activity model to start.",
    };

    /// <summary>Length, codes and error (codec) or segment count (voice activity) of the last run.</summary>
    [ObservableProperty]
    public partial string ResultSummary { get; set; } = "";

    /// <summary>A result is ready.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(PlayCommand), nameof(SaveCommand))]
    public partial bool HasResult { get; set; }

    /// <summary>The reconstruction is playing.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(PlayText))]
    public partial bool IsPlaying { get; set; }

    /// <summary>The text of the play button.</summary>
    public string PlayText => IsPlaying ? "Stop" : "Play";

    /// <summary>A reconstruction is ready to play and export.</summary>
    public bool HasReconstruction => HasResult && IsCodec;

    /// <summary>Why <see cref="RunCommand"/> cannot run, for the hint under the button; empty when it can.</summary>
    public string RunHint => !HasModel ? "Load an audio codec or voice activity model first." : !HasAudio ? "Choose audio first." : "";

    /// <summary>
    /// Takes <paramref name="path"/> as the input: decodes the file and shows its length. Unreadable or unsupported files are refused with a notice.
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

        try
        {
            var audio = await Task.Run(() => Services.Decoder.Decode(path));
            _audio = audio;
            AudioPath = path;
            AudioSummary = $"{AudioFormat.Length(audio.Duration)} · {audio.SampleRate / 1000.0:0.#} kHz · {(audio.Channels == 1 ? "mono" : audio.Channels + " channels")}";
            OnInputChanged();
            return true;
        }
        catch (Exception error) when (error is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
            Notice = $"{Path.GetFileName(path)} cannot be read: {error.Message}";
            return false;
        }
    }

    /// <inheritdoc/>
    protected override void OnModelChanged(ILoadedModel? model)
    {
        var before = Mode;
        ApplyMode();
        if (Mode != before)
        {
            ClearResult();
        }
    }

    /// <inheritdoc/>
    protected override void OnBusyChanged()
    {
        OnPropertyChanged(nameof(IsRunning));
        RunCommand.NotifyCanExecuteChanged();
    }

    /// <inheritdoc/>
    protected override void Restore(RunRecord record)
    {
        Panel.Apply(record.Options);
        ClearResult();
        string audio = Archive.PathOf(record, AudioName);
        string segments = Archive.PathOf(record, SegmentsName);
        if (File.Exists(audio))
        {
            ShowReconstruction(audio, string.Join(" · ", record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}")));
        }
        else if (File.Exists(segments))
        {
            double total = record.Metadata.TryGetValue("seconds", out var seconds) && double.TryParse(seconds, System.Globalization.CultureInfo.InvariantCulture, out double parsed) ? parsed : 1;
            ShowSegments(ParseSegments(File.ReadAllLines(segments)), TimeSpan.FromSeconds(total), string.Join(" · ", record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}")));
        }
        else
        {
            Notice = "The stored result is gone.";
        }
    }

    private void ApplyMode()
    {
        var kind = SelectedModel?.Kind;
        Mode = kind == ModelKinds.AudioCodec ? AudioMode.Codec : kind == ModelKinds.VoiceActivity ? AudioMode.VoiceActivity : AudioMode.None;
        Panel.SetVisible("codebooks", Mode == AudioMode.Codec);
        foreach (string key in (string[])["threshold", "minspeech", "minsilence", "pad"])
        {
            Panel.SetVisible(key, Mode == AudioMode.VoiceActivity);
        }

        OnPropertyChanged(nameof(Mode));
        OnPropertyChanged(nameof(IsCodec));
        OnPropertyChanged(nameof(IsVoiceActivity));
        OnPropertyChanged(nameof(RunText));
        OnPropertyChanged(nameof(ModeDescription));
        OnPropertyChanged(nameof(RunHint));
        RunCommand.NotifyCanExecuteChanged();
    }

    private void OnInputChanged()
    {
        OnPropertyChanged(nameof(HasAudio));
        OnPropertyChanged(nameof(RunHint));
        RunCommand.NotifyCanExecuteChanged();
    }

    private void ClearResult()
    {
        StopPlayback();
        Segments.Clear();
        HasResult = false;
        ResultSummary = "";
        _resultPath = null;
        OnPropertyChanged(nameof(HasReconstruction));
    }

    private bool CanRun() => HasModel && HasAudio && Mode != AudioMode.None;

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

    /// <summary>Runs the chosen model on the input with the current parameters.</summary>
    [RelayCommand(CanExecute = nameof(CanRun))]
    private void Run()
    {
        var model = SelectedModel!;
        var audio = _audio!;
        var settings = Panel.Options;
        var mode = Mode;
        var values = Panel.Snapshot();
        string name = Path.GetFileName(AudioPath);
        Panel.Persist();
        Notice = "";
        ClearResult();
        Queue.Enqueue(name, model.Id, context => RunAsync(model, mode, audio, settings, values, name, context));
        OnPropertyChanged(nameof(IsRunning));
        RunCommand.NotifyCanExecuteChanged();
    }

    private async Task RunAsync(ILoadedModel model, AudioMode mode, AudioBuffer audio, AudioSettings settings, Dictionary<string, string> values, string name, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        RunOutcome outcome = RunOutcome.Completed;
        string? error = null;
        var metadata = new Dictionary<string, string>();
        Dictionary<string, byte[]>? files = null;
        List<SpeechSegment>? segments = null;
        try
        {
            context.Report(null, "Loading the model");
            if (mode == AudioMode.Codec)
            {
                using var lease = await model.AcquireClientAsync<IAudioCodec>(context.CancellationToken).ConfigureAwait(false);
                context.Report(null, "Encoding");
                var codes = await lease.Client.EncodeAsync(audio, settings.Codec, context.CancellationToken).ConfigureAwait(false);
                context.Report(null, "Decoding");
                var decoded = await lease.Client.DecodeAsync(codes, context.CancellationToken).ConfigureAwait(false);
                using var stored = new MemoryStream();
                WavFile.Write(stored, decoded);
                files = new Dictionary<string, byte[]> { [AudioName] = stored.ToArray() };
                var input = audio.ToModelInput(lease.Client.SampleRate).Samples;
                double error2 = 0;
                double energy = 0;
                for (int i = 0; i < Math.Min(input.Length, decoded.Samples.Length); i++)
                {
                    double d = decoded.Samples[i] - input[i];
                    error2 += d * d;
                    energy += (double)input[i] * input[i];
                }

                metadata["codes"] = $"{codes.Codebooks} codebooks x {codes.Frames} frames";
                metadata["bit rate"] = $"{codes.Codebooks * Math.Log2(lease.Client.CodebookSize) * lease.Client.FrameRate / 1000:0.0} kbps";
                metadata["error"] = $"{10 * Math.Log10(Math.Max(error2, 1e-20) / Math.Max(energy, 1e-20)):0.0} dB relative to the input";
            }
            else
            {
                using var lease = await model.AcquireClientAsync<IVoiceActivityDetector>(context.CancellationToken).ConfigureAwait(false);
                context.Report(null, "Listening");
                segments = [.. await lease.Client.DetectAsync(audio, settings.VoiceActivity, context.CancellationToken).ConfigureAwait(false)];
                double speech = segments.Sum(s => s.Duration.TotalSeconds);
                metadata["segments"] = segments.Count.ToString(System.Globalization.CultureInfo.InvariantCulture);
                metadata["speech"] = $"{speech:0.0} s of {audio.Duration.TotalSeconds:0.0} s";
                metadata["seconds"] = audio.Duration.TotalSeconds.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
                files = new Dictionary<string, byte[]>
                {
                    [SegmentsName] = System.Text.Encoding.UTF8.GetBytes(string.Concat(segments.Select(s => FormattableString.Invariant($"{s.Start.TotalSeconds:0.000}\t{s.End.TotalSeconds:0.000}\n")))),
                };
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

        double elapsed = watch.Elapsed.TotalSeconds;
        metadata["time"] = $"{elapsed:0.0} s";
        var record = new RunRecord
        {
            Id = PageArchive.NewId(),
            Title = name,
            Model = model.Id,
            Outcome = outcome,
            Seconds = elapsed,
            Error = error,
            Input = new Dictionary<string, string> { ["file"] = name },
            Options = values,
            Metadata = metadata,
            Preview = string.Join(" · ", metadata.Select(pair => $"{pair.Key}: {pair.Value}")),
            Files = outcome == RunOutcome.Completed && files is not null ? [.. files.Keys] : [],
        };
        if (outcome != RunOutcome.Cancelled)
        {
            Services.Dispatcher.Post(() =>
            {
                Archive.Add(record, outcome == RunOutcome.Completed ? files : null);
                if (outcome != RunOutcome.Completed)
                {
                    return;
                }

                string summary = string.Join(" · ", metadata.Select(pair => $"{pair.Key}: {pair.Value}"));
                if (segments is not null)
                {
                    ShowSegments(segments.Select(s => (s.Start, s.End)).ToList(), audio.Duration, summary);
                }
                else
                {
                    ShowReconstruction(Archive.PathOf(record, AudioName), summary);
                }
            });
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

    private static List<(TimeSpan Start, TimeSpan End)> ParseSegments(IEnumerable<string> lines)
    {
        var result = new List<(TimeSpan, TimeSpan)>();
        foreach (string line in lines)
        {
            var parts = line.Split('\t');
            if (parts.Length == 2 && double.TryParse(parts[0], System.Globalization.CultureInfo.InvariantCulture, out double start) && double.TryParse(parts[1], System.Globalization.CultureInfo.InvariantCulture, out double end))
            {
                result.Add((TimeSpan.FromSeconds(start), TimeSpan.FromSeconds(end)));
            }
        }

        return result;
    }

    private void ShowSegments(IReadOnlyList<(TimeSpan Start, TimeSpan End)> segments, TimeSpan total, string summary)
    {
        Segments.Clear();
        foreach (var (start, end) in segments)
        {
            Segments.Add(new SpeechSegmentViewModel(new SpeechSegment((long)(start.TotalSeconds * 1000), (long)(end.TotalSeconds * 1000), 1000), total));
        }

        ResultSummary = summary;
        HasResult = true;
        OnPropertyChanged(nameof(HasReconstruction));
    }

    private void ShowReconstruction(string path, string summary)
    {
        _resultPath = path;
        ResultSummary = summary;
        HasResult = true;
        IsPlaying = false;
        OnPropertyChanged(nameof(HasReconstruction));
    }

    private bool CanUseResult() => HasResult && IsCodec;

    /// <summary>Plays the reconstruction in the system's audio output, or stops it.</summary>
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

    /// <summary>Saves the reconstruction as a WAV file where the user chooses.</summary>
    [RelayCommand(CanExecute = nameof(CanUseResult))]
    private async Task SaveAsync()
    {
        if (_resultPath is { } path && File.Exists(path))
        {
            await Exporter.SaveAsync(Services.Platform, "Save audio", "reconstruction.wav", await File.ReadAllBytesAsync(path));
        }
    }

    /// <summary>Cancels the queued and running runs and stops playback.</summary>
    [RelayCommand]
    private void Cancel()
    {
        Queue.CancelAll();
        StopPlayback();
    }

    private static ParameterPanelViewModel<AudioSettings> CreateParameters(StudioServices services)
    {
        ParameterViewModel[] parameters =
        [
            new NumberParameterViewModel("codebooks", "Codebooks", "How many residual quantizer stages to keep: more is a higher bit rate and a closer reconstruction.", 1, 32, 1, 8),
            new NumberParameterViewModel("threshold", "Threshold", "A frame is speech when its probability is at least this.", 0.05, 0.95, 0.05, 0.5),
            new NumberParameterViewModel("minspeech", "Minimum speech (ms)", "Shorter segments are dropped.", 0, 2000, 10, 250),
            new NumberParameterViewModel("minsilence", "Minimum silence (ms)", "Silence this long ends a segment.", 0, 2000, 10, 100),
            new NumberParameterViewModel("pad", "Speech padding (ms)", "Every segment is extended by this much on both sides.", 0, 500, 10, 30),
        ];
        ParameterPreset[] presets =
        [
            new("Default", new() { ["codebooks"] = "8", ["threshold"] = "0.5", ["minspeech"] = "250", ["minsilence"] = "100", ["pad"] = "30" }, BuiltIn: true),
            new("Low bit rate", new() { ["codebooks"] = "2" }, BuiltIn: true),
            new("Sensitive", new() { ["threshold"] = "0.3", ["minspeech"] = "100", ["minsilence"] = "200" }, BuiltIn: true),
        ];
        return new ParameterPanelViewModel<AudioSettings>(services.Pages, "audio", parameters, panel => new AudioSettings(
            new AudioCodecOptions { Codebooks = panel.GetInt("codebooks") },
            new VoiceActivityOptions
            {
                Threshold = (float)panel.GetNumber("threshold"),
                MinSpeechDuration = TimeSpan.FromMilliseconds(panel.GetNumber("minspeech")),
                MinSilenceDuration = TimeSpan.FromMilliseconds(panel.GetNumber("minsilence")),
                SpeechPad = TimeSpan.FromMilliseconds(panel.GetNumber("pad")),
            }), presets);
    }
}
