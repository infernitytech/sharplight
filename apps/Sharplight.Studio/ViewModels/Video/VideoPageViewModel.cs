using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Gallery;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.ViewModels.Gallery;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Video;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>
/// The Video page (kind <c>video-generation</c>): a prompt (and negative prompt, and for models that take one a start image) goes to
/// the chosen model's <see cref="IVideoGenerator"/>. The streaming call reports the phases (prompt encoding, denoising steps, decoding);
/// previews and the decoded frames are shown as they arrive, then the finished video plays in the <see cref="Player"/> and can be
/// exported through the <see cref="VideoEncoderCatalog"/> (WebM, MP4 where ffmpeg exists, GIF, APNG). Built from the shared inference blocks.
/// </summary>
internal sealed partial class VideoPageViewModel : InferencePageViewModel
{
    private const string ParametersName = "parameters.json";
    private const string StillName = "first-frame.png";
    private const int PromptTitleLength = 70;
    private const int DefaultSizeMultiple = 16;
    private const int DefaultFrameStep = 4;

    private static readonly ParameterChoice[] s_samplers = [new("", "Model default"), new(Samplers.UniPc, "UniPC"), new(Samplers.Euler, "Euler")];

    private readonly VideoEncoderCatalog _encoders;
    private VideoGenerationDefaults? _defaults;

    /// <summary>The model whose defaults the panel was set to (they are applied once per model, not on every state change of it).</summary>
    private string? _defaultsOf;
    private VideoParametersFile? _last;
    private VideoGenerationResult? _result;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    /// <param name="encoders">The video encoders for export; null uses the ones of <c>AddSharplightVideo()</c> (WebM, MP4 with ffmpeg, GIF, APNG).</param>
    public VideoPageViewModel(StudioServices services, INavigator navigator, VideoEncoderCatalog? encoders = null)
        : base(services, navigator, ModelKinds.VideoGeneration, "video", CreateParameters(services))
    {
        _encoders = encoders ?? new ServiceCollection().AddSharplightVideo().BuildServiceProvider().GetRequiredService<VideoEncoderCatalog>();
        ExportFormats = CreateExportFormats(_encoders);
        Source = new ImageInputViewModel(services.Platform, text => Notice = text, "Start image", "Optional: the first frame of the video (image to video). Drop a file here or choose one.");
        Player = new VideoPlayerViewModel();
        Player.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(VideoPlayerViewModel.HasFrames))
            {
                OnPropertyChanged(nameof(ShowPlaceholder));
            }
        };
        _store = new ImageGalleryStore(services.Pages.GenerationsDirectory("videos"));
        Generations = new ImageGalleryViewModel(_store, services.Platform, services.Dispatcher, OpenStored, ReuseStored, "videos");
    }

    private readonly ImageGalleryStore _store;
    private bool _galleryLoaded;

    /// <summary>Every video ever generated (in the data folder of the storage mode): the first frame as its thumbnail, the video file and a JSON sidecar.</summary>
    public ImageGalleryViewModel Generations { get; }

    /// <inheritdoc/>
    public override Task ActivateAsync()
    {
        if (!_galleryLoaded)
        {
            _galleryLoaded = true;
            _ = Generations.LoadAsync();
        }

        return base.ActivateAsync();
    }

    /// <summary>Shows a stored video's first frame (the file is read and decoded on a worker thread); the video file is exported from the gallery.</summary>
    private void OpenStored(GalleryItemViewModel item)
    {
        var entry = item.Entry;
        _ = Task.Run(() =>
        {
            ImageInput frame;
            try
            {
                frame = ImageFiles.Decode(_store.ReadPng(entry));
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException or InvalidDataException or NotSupportedException)
            {
                Services.Dispatcher.Post(() => Notice = "The video's first frame could not be read: " + error.Message);
                return;
            }

            Services.Dispatcher.Post(() =>
            {
                if (!ReferenceEquals(Generations.SelectedItem, item))
                {
                    return;
                }

                Notice = "";
                _result = null;
                OnPropertyChanged(nameof(HasResult));
                ExportCommand.NotifyCanExecuteChanged();
                Player.Load([frame], entry.FramesPerSecond);
                StepText = "";
                ResultSummary = $"{entry.Model} · size: {entry.Width} × {entry.Height} · frames: {entry.Frames} at {entry.FramesPerSecond.ToString("0.#", CultureInfo.InvariantCulture)} fps · seed: {entry.Seed} · {entry.Created.LocalDateTime:g}"
                    + (entry.Video.Length > 0 ? "" : " · no video file (the first frame only)");
            });
        });
    }

    /// <summary>Fills the prompts, the seed and the parameter panel with a stored video's settings, and chooses its model when it is loaded.</summary>
    private void ReuseStored(GalleryEntry entry)
    {
        Panel.Apply(new Dictionary<string, string>(entry.Options) { ["seed"] = entry.Seed.ToString(CultureInfo.InvariantCulture) });
        Prompt = entry.Prompt;
        NegativePrompt = entry.NegativePrompt;
        if (Services.Runtime.LoadedModels.FirstOrDefault(model => model.Id == entry.Model && model.Kind == ModelKinds.VideoGeneration) is { } model)
        {
            SelectModel(model);
        }

        Notice = "";
    }

    /// <summary>Writes the video, its first frame and its sidecar into the gallery folder (on the worker thread); a full disk must not lose the result.</summary>
    private void StoreInGallery(VideoParametersFile file, GenerationPlan plan, byte[] firstFrame, (string Name, byte[] Bytes)? video, double seconds)
    {
        var entry = new GalleryEntry
        {
            Id = ImageGalleryStore.NewId(),
            Model = file.Model,
            Prompt = file.Prompt,
            NegativePrompt = file.NegativePrompt,
            Seed = file.Seed,
            Steps = plan.Options.Steps ?? 0,
            Guidance = plan.Options.GuidanceScale ?? 0,
            Width = file.Width,
            Height = file.Height,
            Frames = file.Frames,
            FramesPerSecond = file.FramesPerSecond,
            Sampler = plan.Options.Scheduler ?? "",
            Seconds = Math.Round(seconds, 2),
            Options = new Dictionary<string, string>(file.Options),
        };
        try
        {
            string? extension = video is { } encoded ? Path.GetExtension(encoded.Name) : null;
            var stored = _store.Add(extension is null ? entry : entry with { Video = entry.Id + extension }, firstFrame, video?.Bytes);
            Services.Dispatcher.Post(() => Generations.Add(stored));
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Services.Dispatcher.Post(() => Notice = "The video could not be added to the gallery: " + error.Message);
        }
    }

    /// <inheritdoc/>
    public override string Title => "Video";

    /// <inheritdoc/>
    public override string Icon => "IconVideo";

    /// <inheritdoc/>
    public override string KindNoun => "video generation";

    /// <summary>The typed parameter panel (maps to <see cref="VideoGenerationOptions"/>).</summary>
    public ParameterPanelViewModel<VideoGenerationOptions> Panel => (ParameterPanelViewModel<VideoGenerationOptions>)Parameters;

    /// <summary>The start image of an image-to-video run (used when the model <see cref="SupportsImage"/>).</summary>
    public ImageInputViewModel Source { get; }

    /// <summary>The playback of the frames shown.</summary>
    public VideoPlayerViewModel Player { get; }

    /// <summary>The export buttons, one per file format.</summary>
    public IReadOnlyList<VideoExportFormatViewModel> ExportFormats { get; }

    /// <summary>What the video should show.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RunCommand))]
    [NotifyPropertyChangedFor(nameof(RunHint))]
    public partial string Prompt { get; set; } = "";

    /// <summary>What the video should not show.</summary>
    [ObservableProperty]
    public partial string NegativePrompt { get; set; } = "";

    /// <summary>The chosen model takes a start image (image to video).</summary>
    [ObservableProperty]
    public partial bool SupportsImage { get; private set; }

    /// <summary>The phase and step the run is at, or empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasStepText))]
    public partial string StepText { get; set; } = "";

    /// <summary>There is a progress text.</summary>
    public bool HasStepText => StepText.Length > 0;

    /// <summary>The settings and the time of the last result.</summary>
    [ObservableProperty]
    public partial string ResultSummary { get; set; } = "";

    /// <summary>The result of the last run (null before one and while the page shows a restored run).</summary>
    public VideoGenerationResult? Result => _result;

    /// <summary>The page is generating (a run is queued or running).</summary>
    public bool IsGenerating => Queue.IsBusy;

    /// <summary>A finished video can be exported.</summary>
    public bool HasResult => _result is not null;

    /// <summary>Nothing is shown yet.</summary>
    public bool ShowPlaceholder => !Player.HasFrames;

    /// <summary>Why <see cref="RunCommand"/> cannot run, for the hint under the button; empty when it can.</summary>
    public string RunHint => !HasModel ? "Load a video generation model first." : Prompt.Trim().Length == 0 ? "Type a prompt first." : "";

    /// <summary>The latest reading of a model's defaults (for tests).</summary>
    internal Task ModelDefaults { get; private set; } = Task.CompletedTask;

    /// <summary>Rounds <paramref name="value"/> to the nearest multiple of <paramref name="multiple"/> (at least one multiple).</summary>
    /// <param name="value">A width or height.</param>
    /// <param name="multiple">The model's size multiple.</param>
    internal static int SnapSize(int value, int multiple) => Math.Max(multiple, (int)Math.Round(value / (double)multiple) * multiple);

    /// <summary>Rounds <paramref name="value"/> to the nearest frame count of the form 1 + k * <paramref name="step"/>.</summary>
    /// <param name="value">A frame count.</param>
    /// <param name="step">The model's frame step.</param>
    internal static int SnapFrames(int value, int step) => 1 + Math.Max(0, (int)Math.Round((value - 1) / (double)step)) * step;

    /// <inheritdoc/>
    protected override void OnModelChanged(ILoadedModel? model)
    {
        OnPropertyChanged(nameof(RunHint));
        RunCommand.NotifyCanExecuteChanged();
        if (model is { Kind: ModelKinds.VideoGeneration, State: ResidentState.Ready or ResidentState.Busy } && model.Id != _defaultsOf)
        {
            _defaultsOf = model.Id;
            ModelDefaults = ApplyModelDefaultsAsync(model);
        }
    }

    /// <summary>
    /// Sets size, frames, frame rate, steps, guidance and sampler to the model's defaults (Wan 2.1: 832 × 480, 81 frames at 16 fps,
    /// 50 steps), prefills the negative prompt when the model has a stock one, and shows the start image only when the model takes one.
    /// </summary>
    private async Task ApplyModelDefaultsAsync(ILoadedModel model)
    {
        VideoGenerationDefaults? defaults;
        try
        {
            using var lease = await model.AcquireClientAsync<IVideoGenerator>(CancellationToken.None).ConfigureAwait(false);
            defaults = lease.Client.GetService(typeof(VideoGenerationDefaults)) as VideoGenerationDefaults;
        }
        catch (Exception error) when (error is InvalidOperationException or NotSupportedException or ObjectDisposedException)
        {
            return;
        }

        if (defaults is null)
        {
            return;
        }

        Services.Dispatcher.Post(() =>
        {
            if (SelectedModel?.Id != model.Id)
            {
                return;
            }

            _defaults = defaults;
            Panel.Apply(new Dictionary<string, string>
            {
                ["width"] = defaults.Width.ToString(CultureInfo.InvariantCulture),
                ["height"] = defaults.Height.ToString(CultureInfo.InvariantCulture),
                ["frames"] = defaults.Frames.ToString(CultureInfo.InvariantCulture),
                ["fps"] = defaults.FramesPerSecond.ToString(CultureInfo.InvariantCulture),
                ["steps"] = defaults.Steps.ToString(CultureInfo.InvariantCulture),
                ["guidance"] = defaults.GuidanceScale.ToString(CultureInfo.InvariantCulture),
                ["sampler"] = Samplers.Flow.Contains(defaults.Sampler) ? defaults.Sampler : "",
            });
            SupportsImage = defaults.SupportsImage;
            if (!defaults.SupportsImage)
            {
                Source.Clear();
            }

            if (!string.IsNullOrEmpty(defaults.NegativePrompt) && NegativePrompt.Length == 0)
            {
                NegativePrompt = defaults.NegativePrompt;
            }
        });
    }

    /// <inheritdoc/>
    protected override void OnBusyChanged()
    {
        OnPropertyChanged(nameof(IsGenerating));
        RunCommand.NotifyCanExecuteChanged();
    }

    /// <inheritdoc/>
    protected override void Restore(RunRecord record)
    {
        string parametersPath = Archive.PathOf(record, ParametersName);
        if (!File.Exists(parametersPath))
        {
            Notice = "The stored video is gone.";
            return;
        }

        var file = PageJson.Read(parametersPath, StudioJsonContext.Default.VideoParametersFile);
        Notice = "";
        Panel.Apply(file.Options.Count > 0 ? file.Options : record.Options);
        Prompt = file.Prompt;
        NegativePrompt = file.NegativePrompt;
        _last = file;
        _result = null;
        OnPropertyChanged(nameof(HasResult));
        ExportCommand.NotifyCanExecuteChanged();
        string stillPath = Archive.PathOf(record, StillName);
        if (File.Exists(stillPath))
        {
            Player.Load([ImageFiles.Decode(File.ReadAllBytes(stillPath))], file.FramesPerSecond);
        }
        else
        {
            Player.Clear();
        }

        StepText = "";
        ResultSummary = string.Join(" · ", record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}"));
    }

    private bool CanRun() => HasModel && Prompt.Trim().Length > 0;

    /// <summary>Generates the video with the chosen model and the current parameters.</summary>
    [RelayCommand(CanExecute = nameof(CanRun))]
    private void Run()
    {
        if (!SeedParser.TryRead(Panel.GetText("seed"), out long? seed))
        {
            Notice = "The seed must be a whole number (leave it empty for a random one).";
            return;
        }

        var model = SelectedModel!;
        var options = SnapToModel(Panel.Options);
        var values = Panel.Snapshot();
        values["width"] = options.Width!.Value.ToString(CultureInfo.InvariantCulture);
        values["height"] = options.Height!.Value.ToString(CultureInfo.InvariantCulture);
        values["frames"] = options.Frames!.Value.ToString(CultureInfo.InvariantCulture);
        var image = SupportsImage ? Source.Image : null;
        var plan = new GenerationPlan(Prompt.Trim(), NegativePrompt.Trim(), options, seed, image, values, Panel.GetBool("preview"));
        string title = plan.Prompt.Length > PromptTitleLength ? plan.Prompt[..PromptTitleLength] + "…" : plan.Prompt;
        Panel.Persist();
        Notice = "";
        StepText = "";
        Player.Clear();
        Queue.Enqueue(title, model.Id, context => GenerateAsync(model, plan, title, context));
        OnPropertyChanged(nameof(IsGenerating));
        RunCommand.NotifyCanExecuteChanged();
    }

    /// <summary>The options with width and height on the model's size multiple and the frame count on 1 + k * frame step.</summary>
    private VideoGenerationOptions SnapToModel(VideoGenerationOptions options)
    {
        int multiple = _defaults?.SizeMultiple ?? DefaultSizeMultiple;
        int step = _defaults?.FrameStep ?? DefaultFrameStep;
        return options with
        {
            Width = SnapSize(options.Width ?? DefaultSizeMultiple, multiple),
            Height = SnapSize(options.Height ?? DefaultSizeMultiple, multiple),
            Frames = SnapFrames(options.Frames ?? 1, step),
        };
    }

    private async Task GenerateAsync(ILoadedModel model, GenerationPlan plan, string title, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        VideoGenerationResult? result = null;
        var received = new List<ImageInput>();
        RunOutcome outcome = RunOutcome.Completed;
        string? error = null;
        try
        {
            context.Report(null, "Loading the model");
            using var lease = await model.AcquireClientAsync<IVideoGenerator>(context.CancellationToken).ConfigureAwait(false);
            var request = new VideoGenerationRequest
            {
                Prompt = plan.Prompt,
                NegativePrompt = plan.NegativePrompt.Length > 0 ? plan.NegativePrompt : null,
                Image = plan.Image,
            };
            var options = plan.Options with { Seed = plan.Seed, PreviewInterval = plan.Preview ? Math.Max(1, (plan.Options.Steps ?? 20) / 5) : 0 };
            context.Report(0, "Starting");
            await foreach (var update in lease.Client.GenerateStreamingAsync(request, options, context.CancellationToken).ConfigureAwait(false))
            {
                ShowProgress(context, update, received, plan.Options.Frames ?? 0);
                result = update.Result ?? result;
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
        if (outcome == RunOutcome.Completed && result is null)
        {
            outcome = RunOutcome.Failed;
            error = "The model returned no video.";
        }

        if (outcome != RunOutcome.Completed)
        {
            Services.Dispatcher.Post(() =>
            {
                Player.Clear();
                StepText = "";
            });
            if (outcome == RunOutcome.Failed)
            {
                Services.Dispatcher.Post(() => Notice = error!);
                Archived(NewRecord(title, model, plan, outcome, seconds, error, null, [], []));
                throw new InvalidOperationException(error);
            }

            throw new OperationCanceledException(context.CancellationToken);
        }

        await FinishAsync(model, plan, title, result!, seconds, context.CancellationToken).ConfigureAwait(false);
    }

    private void ShowProgress(RunContext context, VideoGenerationUpdate update, List<ImageInput> received, int requestedFrames)
    {
        string text;
        double? fraction;
        switch (update.Phase)
        {
            case VideoGenerationPhase.Conditioning:
                text = "Encoding the prompt";
                fraction = 0.02;
                break;
            case VideoGenerationPhase.Denoising:
                text = $"Denoising: step {update.Step} of {update.TotalSteps}";
                fraction = update.TotalSteps > 0 ? 0.05 + 0.85 * update.Step / update.TotalSteps : null;
                break;
            case VideoGenerationPhase.Decoding:
                text = $"Decoding: {received.Count + (update.Frames?.Count ?? 0)} frames";
                fraction = requestedFrames > 0 ? 0.9 + 0.1 * Math.Min(1, (received.Count + (update.Frames?.Count ?? 0)) / (double)requestedFrames) : null;
                break;
            default:
                text = "Done";
                fraction = 1;
                break;
        }

        context.Report(fraction, text);
        ImageInput[]? shown = null;
        if (update.Frames is { Count: > 0 } decoded)
        {
            received.AddRange(decoded);
            shown = [.. received];
        }
        else if (update.Preview is { Count: > 0 } preview && received.Count == 0)
        {
            shown = [.. preview];
        }

        Services.Dispatcher.Post(() =>
        {
            StepText = text;
            if (shown is not null)
            {
                Player.Follow(shown);
            }
        });
    }

    private async Task FinishAsync(ILoadedModel model, GenerationPlan plan, string title, VideoGenerationResult finished, double seconds, CancellationToken cancellationToken)
    {
        var first = finished.Frames[0];
        var file = new VideoParametersFile
        {
            Model = model.Id,
            Prompt = plan.Prompt,
            NegativePrompt = plan.NegativePrompt,
            Seed = finished.Seed,
            Width = first.Width,
            Height = first.Height,
            Frames = finished.Frames.Count,
            FramesPerSecond = finished.FramesPerSecond,
            ImageToVideo = plan.Image is not null,
            Options = new Dictionary<string, string>(plan.Values) { ["seed"] = finished.Seed.ToString(CultureInfo.InvariantCulture) },
        };
        var metadata = new Dictionary<string, string>
        {
            ["size"] = $"{first.Width} × {first.Height}",
            ["frames"] = $"{file.Frames} at {file.FramesPerSecond.ToString("0.#", CultureInfo.InvariantCulture)} fps",
            ["steps"] = (plan.Options.Steps ?? 0).ToString(CultureInfo.InvariantCulture),
            ["seed"] = file.Seed.ToString(CultureInfo.InvariantCulture),
            ["time"] = $"{seconds:0.0} s",
        };
        if (file.ImageToVideo)
        {
            metadata["mode"] = "image to video";
        }

        string summary = string.Join(" · ", metadata.Select(pair => $"{pair.Key}: {pair.Value}"));
        Services.Dispatcher.Post(() =>
        {
            _last = file;
            _result = finished;
            Player.Load(finished.Frames, finished.FramesPerSecond);
            StepText = "";
            ResultSummary = summary;
            OnPropertyChanged(nameof(Result));
            OnPropertyChanged(nameof(HasResult));
            ExportCommand.NotifyCanExecuteChanged();
        });

        var files = new Dictionary<string, byte[]>
        {
            [StillName] = ImageFiles.ToPng(first),
            [ParametersName] = JsonSerializer.SerializeToUtf8Bytes(file, StudioJsonContext.Default.VideoParametersFile),
        };
        if (await TryEncodeAsync(finished, cancellationToken).ConfigureAwait(false) is { } video)
        {
            files[video.Name] = video.Bytes;
        }

        StoreInGallery(file, plan, files[StillName], files.Keys.FirstOrDefault(name => name.StartsWith("video.", StringComparison.Ordinal)) is { } name ? (name, files[name]) : null, seconds);
        Archived(NewRecord(title, model, plan, RunOutcome.Completed, seconds, null, file, [.. files.Keys], metadata), files);
    }

    /// <summary>The video as a file for the history (WebM, else GIF), or null when no encoder produced one.</summary>
    private async Task<(string Name, byte[] Bytes)?> TryEncodeAsync(VideoGenerationResult result, CancellationToken cancellationToken)
    {
        foreach (string format in new[] { VideoFormats.WebM, VideoFormats.Gif })
        {
            if (!_encoders.TryFind(format, out var encoder))
            {
                continue;
            }

            try
            {
                using var stream = new MemoryStream();
                await encoder.EncodeAsync(result.ToClip(), stream, null, cancellationToken).ConfigureAwait(false);
                return ($"video.{encoder.FileExtensions[0].TrimStart('.')}", stream.ToArray());
            }
            catch (Exception error) when (error is NotSupportedException or IOException or PlatformNotSupportedException or ArgumentException)
            {
                continue;
            }
        }

        return null;
    }

    private RunRecord NewRecord(string title, ILoadedModel model, GenerationPlan plan, RunOutcome outcome, double seconds, string? error, VideoParametersFile? file,
        List<string> fileNames, Dictionary<string, string> metadata)
    {
        var input = new Dictionary<string, string> { ["prompt"] = plan.Prompt };
        if (plan.NegativePrompt.Length > 0)
        {
            input["negative prompt"] = plan.NegativePrompt;
        }

        if (plan.Image is not null)
        {
            input["start image"] = $"{plan.Image.Width} × {plan.Image.Height}";
        }

        return new RunRecord
        {
            Id = PageArchive.NewId(),
            Title = title,
            Model = model.Id,
            Outcome = outcome,
            Seconds = seconds,
            Error = error,
            Input = input,
            Options = file?.Options ?? plan.Values,
            Metadata = metadata,
            Preview = plan.NegativePrompt.Length > 0 ? $"Without: {plan.NegativePrompt}" : null,
            Files = fileNames,
        };
    }

    /// <summary>Puts the last result's seed into the parameters, so the next run starts from the same noise.</summary>
    [RelayCommand]
    private void ReuseSeed()
    {
        if (_last is { } file)
        {
            Panel.Parameters.OfType<TextParameterViewModel>().First(parameter => parameter.Key == "seed").Value = file.Seed.ToString(CultureInfo.InvariantCulture);
        }
    }

    /// <summary>Takes the prompt and negative prompt of the shown result back into the boxes.</summary>
    [RelayCommand]
    private void ReusePrompt()
    {
        if (_last is { } file)
        {
            Prompt = file.Prompt;
            NegativePrompt = file.NegativePrompt;
        }
    }

    /// <summary>Cancels the queued and running generations.</summary>
    [RelayCommand]
    private void Cancel() => Queue.CancelAll();

    private bool CanExport(VideoExportFormatViewModel? format) => HasResult && format is { IsAvailable: true };

    /// <summary>Asks where to save the video and writes it in <paramref name="format"/>.</summary>
    /// <param name="format">The file format chosen.</param>
    [RelayCommand(CanExecute = nameof(CanExport))]
    private async Task ExportAsync(VideoExportFormatViewModel? format)
    {
        if (format is null || _result is null)
        {
            return;
        }

        var filter = new FileFilter(format.Label, [$"*{format.Extension}"]);
        if (await Services.Platform.PickSaveFileAsync("Export video", $"sharplight-{_result.Seed}{format.Extension}", filter) is { } path)
        {
            await ExportToAsync(path);
        }
    }

    /// <summary>Encodes the shown video to <paramref name="path"/> (the format follows the extension) off the UI thread.</summary>
    /// <param name="path">The file to write.</param>
    internal async Task ExportToAsync(string path)
    {
        if (_result is not { } result)
        {
            return;
        }

        try
        {
            var clip = result.ToClip();
            await Task.Run(() => _encoders.SaveAsync(clip, path)).ConfigureAwait(true);
            Notice = "";
            StepText = $"Saved {Path.GetFileName(path)}";
        }
        catch (Exception failure) when (failure is NotSupportedException or IOException or PlatformNotSupportedException or UnauthorizedAccessException)
        {
            Notice = $"The video could not be exported: {failure.Message}";
        }
    }

    private static VideoExportFormatViewModel[] CreateExportFormats(VideoEncoderCatalog encoders)
    {
        bool Has(string format) => encoders.TryFind(format, out _);
        return
        [
            new(VideoFormats.Mp4, "MP4", ".mp4", Has(VideoFormats.Mp4), Has(VideoFormats.Mp4) ? "H.264 video through ffmpeg" : "Needs ffmpeg on the PATH (H.264 is not bundled)"),
            new(VideoFormats.WebM, "WebM", ".webm", Has(VideoFormats.WebM), Has(VideoFormats.WebM) ? "VP9 video" : "The native video library is not available on this system"),
            new(VideoFormats.Gif, "GIF", ".gif", Has(VideoFormats.Gif), "Animated GIF (256 colors per frame)"),
            new(VideoFormats.Apng, "APNG", ".apng", Has(VideoFormats.Apng), "Animated PNG (lossless)"),
        ];
    }

    private static ParameterPanelViewModel<VideoGenerationOptions> CreateParameters(StudioServices services)
    {
        ParameterViewModel[] parameters =
        [
            new NumberParameterViewModel("width", "Width", "Pixels; rounded to the model's size multiple (16 for Wan 2.1, 32 for Wan 2.2 TI2V).", 16, 1920, 16, 512),
            new NumberParameterViewModel("height", "Height", "Pixels; rounded to the model's size multiple.", 16, 1920, 16, 320),
            new NumberParameterViewModel("frames", "Frames", "Number of frames; rounded to 1 + 4k (other counts are rounded down by the model).", 1, 161, 4, 33),
            new NumberParameterViewModel("fps", "Frames per second", "Playback speed of the result.", 1, 60, 1, 16),
            new NumberParameterViewModel("steps", "Steps", "Denoising steps: more are slower and finer.", 1, 100, 1, 30),
            new NumberParameterViewModel("guidance", "Guidance", "How closely the video follows the prompt (classifier-free guidance).", 0, 20, 0.5, 5),
            new ChoiceParameterViewModel("sampler", "Sampler", "The scheduler that steps the noise away.", s_samplers),
            new TextParameterViewModel("seed", "Seed", "Empty picks a random seed; a number repeats the video.", "Random"),
            new BoolParameterViewModel("preview", "Live preview", "Show the video taking shape while it is generated.", defaultValue: true),
        ];
        return new ParameterPanelViewModel<VideoGenerationOptions>(services.Pages, "video", parameters, panel => new VideoGenerationOptions
        {
            Width = panel.GetInt("width"),
            Height = panel.GetInt("height"),
            Frames = panel.GetInt("frames"),
            FramesPerSecond = panel.GetNumber("fps"),
            Steps = panel.GetInt("steps"),
            GuidanceScale = (float)panel.GetNumber("guidance"),
            Scheduler = panel.GetChoice("sampler") is { Length: > 0 } sampler ? sampler : null,
        });
    }

    /// <summary>Everything one run needs, read from the page when it starts.</summary>
    private sealed record GenerationPlan(string Prompt, string NegativePrompt, VideoGenerationOptions Options, long? Seed, ImageInput? Image,
        Dictionary<string, string> Values, bool Preview);
}
