using System.Diagnostics;
using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.ImageGeneration;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Audio;
using Sharplight.Studio.ViewModels.Inference;

namespace Sharplight.Studio.ViewModels.Chat;

/// <summary>
/// What the chat does beyond text, shared by its conversations: dictation (the microphone, transcribed by a resident speech-recognition
/// model into the prompt), reading answers aloud (a resident speech-synthesis model, played while it is generated), and pictures (a
/// resident image-generation model behind the <c>/image</c> command). Every model is leased from the runtime like on the pages of its kind
/// and every call runs off the UI thread. Which model serves each purpose is a setting (<see cref="VoiceSettings"/>).
/// </summary>
internal sealed partial class ChatMediaViewModel : ObservableObject
{
    private readonly StudioServices _services;
    private IRecording? _recording;
    private CancellationTokenSource? _speech;

    public ChatMediaViewModel(StudioServices services)
    {
        _services = services;
        var settings = services.Settings.Current.Voice;
        Recognizers = Prefer(new ModelPickerViewModel(services, ModelKinds.SpeechRecognition), settings.RecognitionModel,
            id => services.Settings.Update(current => current with { Voice = current.Voice with { RecognitionModel = id } }));
        Synthesizers = Prefer(new ModelPickerViewModel(services, ModelKinds.SpeechSynthesis), settings.SynthesisModel,
            id => services.Settings.Update(current => current with { Voice = current.Voice with { SynthesisModel = id } }));
        Painters = Prefer(new ModelPickerViewModel(services, ModelKinds.ImageGeneration), settings.ImageModel,
            id => services.Settings.Update(current => current with { Voice = current.Voice with { ImageModel = id } }));
        ReadAloud = settings.ReadAloud;
    }

    /// <summary>The resident speech-recognition models and the chosen one (dictation).</summary>
    public ModelPickerViewModel Recognizers { get; }

    /// <summary>The resident speech-synthesis models and the chosen one (reading answers aloud).</summary>
    public ModelPickerViewModel Synthesizers { get; }

    /// <summary>The resident image-generation models and the chosen one (the <c>/image</c> command).</summary>
    public ModelPickerViewModel Painters { get; }

    /// <summary>Read every finished answer aloud.</summary>
    [ObservableProperty]
    public partial bool ReadAloud { get; set; }

    /// <summary>The microphone is recording.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DictationText))]
    public partial bool IsRecording { get; set; }

    /// <summary>A recording is being transcribed.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DictationText))]
    public partial bool IsTranscribing { get; set; }

    /// <summary>An answer is being read aloud.</summary>
    [ObservableProperty]
    public partial bool IsSpeaking { get; set; }

    /// <summary>A speech-recognition model is resident, so the microphone button is offered.</summary>
    public bool CanDictate => Recognizers.HasModels;

    /// <summary>A speech-synthesis model is resident, so answers can be read aloud.</summary>
    public bool CanSpeak => Synthesizers.HasModels;

    /// <summary>An image-generation model is resident, so the <c>/image</c> command works.</summary>
    public bool CanPaint => Painters.HasModels;

    /// <summary>The label of the microphone button.</summary>
    public string DictationText => IsTranscribing ? "Transcribing" : IsRecording ? "Stop" : "Dictate";

    partial void OnReadAloudChanged(bool value)
    {
        if (_services.Settings.Current.Voice.ReadAloud != value)
        {
            _services.Settings.Update(current => current with { Voice = current.Voice with { ReadAloud = value } });
        }

        if (!value)
        {
            StopSpeaking();
        }
    }

    /// <summary>
    /// Starts recording the microphone, or stops it and returns what the recognition model made of it.
    /// </summary>
    /// <returns><c>null</c> when recording started or nothing could be done (see <paramref name="notice"/>), else the transcript (possibly empty).</returns>
    /// <param name="notice">Receives a problem to show the user, or null.</param>
    public async Task<string?> ToggleDictationAsync(Action<string?> notice)
    {
        if (IsTranscribing)
        {
            return null;
        }

        if (_recording is { } running)
        {
            return await FinishDictationAsync(running, notice);
        }

        if (Recognizers.SelectedModel is null)
        {
            notice("Load a speech-to-text model to dictate (Models, then Library).");
            return null;
        }

        if (!_services.Recorder.IsAvailable)
        {
            notice(_services.Recorder.Description);
            return null;
        }

        notice(null);
        try
        {
            string path = RecordingPath;
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            _recording = _services.Recorder.Start(path);
            IsRecording = true;
        }
        catch (Exception error) when (error is IOException or InvalidOperationException or System.ComponentModel.Win32Exception or NotSupportedException)
        {
            notice($"The microphone could not be recorded: {error.Message}");
        }

        return null;
    }

    /// <summary>Stops a recording in progress without transcribing it.</summary>
    public async Task CancelDictationAsync()
    {
        if (_recording is { } running)
        {
            _recording = null;
            IsRecording = false;
            await running.StopAsync();
            running.Dispose();
        }
    }

    /// <summary>Reads <paramref name="markdown"/> aloud with the chosen synthesis model, playing the audio while it is generated; replaces what is being read.</summary>
    /// <param name="markdown">The answer as shown in the chat (markup is not spoken).</param>
    /// <param name="notice">Receives a problem to show the user, or null.</param>
    public async Task SpeakAsync(string markdown, Action<string?> notice)
    {
        StopSpeaking();
        string text = SpeakableText.From(markdown);
        if (text.Length == 0)
        {
            return;
        }

        if (Synthesizers.SelectedModel?.Model is not { } model)
        {
            notice("Load a text-to-speech model to read answers aloud.");
            return;
        }

        if (!_services.Player.IsAvailable)
        {
            notice(_services.Player.Description);
            return;
        }

        var speech = new CancellationTokenSource();
        _speech = speech;
        IsSpeaking = true;
        try
        {
            await Task.Run(async () =>
            {
                using var lease = await model.AcquireClientAsync<ISpeechSynthesizer>(speech.Token).ConfigureAwait(false);
                await _services.Player.PlayStreamAsync(lease.Client.SynthesizeStreamingAsync(text, null, speech.Token), speech.Token).ConfigureAwait(false);
            }, CancellationToken.None);
        }
        catch (OperationCanceledException)
        {
            // Stopped by the user or replaced by another answer.
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            notice($"The answer could not be read aloud: {error.GetBaseException().Message}");
        }
        finally
        {
            if (ReferenceEquals(_speech, speech))
            {
                _speech = null;
                IsSpeaking = false;
            }

            speech.Dispose();
        }
    }

    /// <summary>Stops reading aloud.</summary>
    public void StopSpeaking() => _speech?.Cancel();

    /// <summary>Generates one picture for <paramref name="prompt"/> with the chosen image model.</summary>
    /// <param name="prompt">What to draw.</param>
    /// <param name="cancellationToken">Stops generation between denoising steps.</param>
    /// <exception cref="InvalidOperationException">No image-generation model is resident.</exception>
    public async Task<GeneratedPicture> GenerateImageAsync(string prompt, CancellationToken cancellationToken)
    {
        var model = Painters.SelectedModel?.Model ?? throw new InvalidOperationException("Load an image-generation model to create pictures (Models, then Library).");
        return await Task.Run(async () =>
        {
            var watch = Stopwatch.StartNew();
            using var lease = await model.AcquireClientAsync<IImageGenerator>(cancellationToken).ConfigureAwait(false);
            ImageGenerationResult? result = null;
            await foreach (var update in lease.Client.GenerateStreamingAsync(new ImageGenerationRequest { Prompt = prompt }, new ImageGenerationOptions(), cancellationToken).ConfigureAwait(false))
            {
                result = update.Result ?? result;
            }

            var image = result?.Images.FirstOrDefault() ?? throw new InvalidOperationException($"{model.Id} returned no image.");
            return new GeneratedPicture(PngCodec.Encode(image), model.Id, result!.Seed, image.Width, image.Height, watch.Elapsed.TotalSeconds);
        }, CancellationToken.None).ConfigureAwait(false);
    }

    private async Task<string?> FinishDictationAsync(IRecording running, Action<string?> notice)
    {
        _recording = null;
        IsRecording = false;
        await running.StopAsync();
        running.Dispose();
        var model = Recognizers.SelectedModel?.Model;
        if (model is null)
        {
            notice("The speech-to-text model is gone.");
            return null;
        }

        IsTranscribing = true;
        try
        {
            string path = RecordingPath;
            return await Task.Run(async () =>
            {
                var audio = _services.Decoder.Decode(path);
                using var lease = await model.AcquireClientAsync<ISpeechRecognizer>(CancellationToken.None).ConfigureAwait(false);
                var transcript = await lease.Client.TranscribeAsync(audio, null, CancellationToken.None).ConfigureAwait(false);
                return transcript.Text.Trim();
            }, CancellationToken.None);
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            notice($"The recording could not be transcribed: {error.GetBaseException().Message}");
            return null;
        }
        finally
        {
            IsTranscribing = false;
        }
    }

    private string RecordingPath => Path.Combine(_services.Pages.DirectoryFor("chat"), "recordings", "recording.wav");

    private ModelPickerViewModel Prefer(ModelPickerViewModel picker, string? preferred, Action<string> remember)
    {
        picker.PreferredId = preferred;
        if (preferred is not null && picker.Models.FirstOrDefault(option => option.Id == preferred) is { } option)
        {
            picker.SelectedModel = option;
        }

        picker.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(ModelPickerViewModel.SelectedModel) && !picker.IsRefreshing && picker.SelectedModel is { } chosen)
            {
                picker.PreferredId = chosen.Id;
                remember(chosen.Id);
            }
            else if (args.PropertyName == nameof(ModelPickerViewModel.HasModels))
            {
                OnPropertyChanged(nameof(CanDictate));
                OnPropertyChanged(nameof(CanSpeak));
                OnPropertyChanged(nameof(CanPaint));
            }
        };
        return picker;
    }
}
