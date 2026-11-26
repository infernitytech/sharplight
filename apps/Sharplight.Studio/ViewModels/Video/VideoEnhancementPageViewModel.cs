using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Video;
using Sharplight.VideoEnhancement;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>
/// The Enhance page (kind <c>video-enhancement</c>): a video or a still image goes to the chosen model's <see cref="IVideoEnhancer"/>
/// with an output size (a target height or an enlargement), frames per batch, colour correction and seed. A video is streamed: decoded
/// frame by frame (<see cref="VideoDecoderCatalog"/>) into the enhancer, and each finished batch is encoded into the output video (MP4
/// through the bundled FFmpeg, else WebM or Matroska, with the input's sound track) and shown in the live comparison, so a clip of any
/// length runs in bounded memory; progress shows the frames, the playback time and the time left. A still image is written as a PNG.
/// Results land in the page's output folder, from where they can be opened or saved elsewhere; a cancelled or failed run keeps the video
/// finished so far. Without the bundled FFmpeg only images can be enhanced. Built from the shared inference blocks, like every task page;
/// what the page offers follows the model's <see cref="VideoEnhancementDefaults"/>.
/// </summary>
internal sealed partial class VideoEnhancementPageViewModel : InferencePageViewModel
{
    private const string ResultStillName = "result.png";
    private const string OutputMetadataKey = "file";
    private const int MaxOutputWidth = 7680;
    private const int MaxOutputHeight = 4320;
    private const int MinOutputSide = 16;

    /// <summary>Video file types recognized even when no decoder is available (to say why they cannot be read).</summary>
    private static readonly string[] s_videoExtensions = [".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".wmv", ".flv"];

    private static readonly ParameterChoice[] s_colorCorrections =
    [
        new("", "Model default"),
        new(nameof(VideoColorCorrection.Wavelet), "Wavelet (keep the input's colours)"),
        new(nameof(VideoColorCorrection.AdaIn), "AdaIN (match brightness and contrast)"),
        new(nameof(VideoColorCorrection.None), "None (the model's output)"),
    ];

    private readonly VideoEncoderCatalog _encoders;
    private readonly VideoDecoderCatalog _decoders;

    /// <summary>The model whose defaults the panel was set to (applied once per model).</summary>
    private string? _defaultsOf;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    /// <param name="encoders">The video encoders for the result; null uses the ones of <c>AddSharplightVideo()</c>.</param>
    /// <param name="decoders">The video decoders for the input; null uses the one of <c>AddSharplightVideo()</c>.</param>
    public VideoEnhancementPageViewModel(StudioServices services, INavigator navigator, VideoEncoderCatalog? encoders = null, VideoDecoderCatalog? decoders = null)
        : base(services, navigator, ModelKinds.VideoEnhancement, "enhance", CreateParameters(services))
    {
        if (encoders is null || decoders is null)
        {
            ServiceProvider video = new ServiceCollection().AddSharplightVideo().BuildServiceProvider();
            encoders ??= video.GetRequiredService<VideoEncoderCatalog>();
            decoders ??= video.GetRequiredService<VideoDecoderCatalog>();
        }

        _encoders = encoders;
        _decoders = decoders;
        Player = new VideoPlayerViewModel(services.Player);
        Player.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(VideoPlayerViewModel.HasFrames))
            {
                OnPropertyChanged(nameof(ShowPlaceholder));
                OnPropertyChanged(nameof(ShowPlayer));
            }
        };
        Compare = new EnhancementComparisonViewModel(services.Dispatcher);
        Compare.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(EnhancementComparisonViewModel.HasFrames))
            {
                OnPropertyChanged(nameof(ShowPlaceholder));
                OnPropertyChanged(nameof(ShowPlayer));
            }
        };
        foreach (var parameter in Panel.Parameters)
        {
            parameter.PropertyChanged += OnParameterChanged;
        }
    }

    /// <inheritdoc/>
    public override string Title => "Enhance";

    /// <inheritdoc/>
    public override string Icon => "IconUpscale";

    /// <inheritdoc/>
    public override string KindNoun => "video enhancement";

    /// <summary>The typed parameter panel (maps to <see cref="VideoEnhancementOptions"/>).</summary>
    public ParameterPanelViewModel<VideoEnhancementOptions> Panel => (ParameterPanelViewModel<VideoEnhancementOptions>)Parameters;

    /// <summary>The playback of the input (and of a restored result's still) while there is nothing to compare.</summary>
    public VideoPlayerViewModel Player { get; }

    /// <summary>The before/after comparison of the frames finished so far (live while enhancing, the whole clip afterwards).</summary>
    public EnhancementComparisonViewModel Compare { get; }

    /// <summary>The input (or a restored still) is shown: there is nothing to compare.</summary>
    public bool ShowPlayer => Player.HasFrames && !Compare.HasFrames;

    /// <summary>Videos can be read (the bundled FFmpeg libraries are present); without them only images can be enhanced.</summary>
    public bool CanReadVideos => _decoders.IsAvailable;

    /// <summary>Why videos cannot be read, for the notice on the page; empty when they can.</summary>
    public string VideoSupportNotice => CanReadVideos
        ? ""
        : "Videos cannot be read: the FFmpeg libraries Studio ships with are missing from this installation (reinstall Studio to restore them). Images can still be enhanced.";

    /// <summary>The chosen input, or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasInput), nameof(InputPath), nameof(InputSummary), nameof(OutputSizeText), nameof(RunHint))]
    [NotifyCanExecuteChangedFor(nameof(RunCommand))]
    public partial EnhancementInput? Input { get; private set; }

    /// <summary>An input is chosen.</summary>
    public bool HasInput => Input is not null;

    /// <summary>The input file, or empty.</summary>
    public string InputPath => Input?.Path ?? "";

    /// <summary>What the input is (name, size, frames), or a prompt to choose one.</summary>
    public string InputSummary => Input is not { } input
        ? "Choose a video or an image."
        : input.IsVideo
            ? $"{Path.GetFileName(input.Path)} · video · {input.Width} × {input.Height} · {input.Frames} frames at {input.FramesPerSecond.ToString("0.##", CultureInfo.InvariantCulture)} fps"
            : $"{Path.GetFileName(input.Path)} · image · {input.Width} × {input.Height}";

    /// <summary>The output size the current parameters give the input, or empty without an input.</summary>
    public string OutputSizeText => OutputSize() is { } size ? $"Output: {size.Width} × {size.Height}" : "";

    /// <summary>The phase and batch the run is at, or empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasStepText))]
    public partial string StepText { get; set; } = "";

    /// <summary>There is a progress text.</summary>
    public bool HasStepText => StepText.Length > 0;

    /// <summary>The settings and the time of the shown result.</summary>
    [ObservableProperty]
    public partial string ResultSummary { get; set; } = "";

    /// <summary>The file the shown result was written to, or empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasResultFile))]
    [NotifyCanExecuteChangedFor(nameof(OpenResultCommand), nameof(SaveResultCommand))]
    public partial string ResultPath { get; private set; } = "";

    /// <summary>The shown result has a file that can be opened or saved.</summary>
    public bool HasResultFile => ResultPath.Length > 0 && File.Exists(ResultPath);

    /// <summary>The page is enhancing (a run is queued or running).</summary>
    public bool IsEnhancing => Queue.IsBusy;

    /// <summary>Nothing is shown yet.</summary>
    public bool ShowPlaceholder => !Player.HasFrames && !Compare.HasFrames;

    /// <summary>Why <see cref="RunCommand"/> cannot run, for the hint under the button; empty when it can.</summary>
    public string RunHint => InvalidReason() ?? "";

    /// <summary>The latest reading of a model's defaults (for tests).</summary>
    internal Task ModelDefaults { get; private set; } = Task.CompletedTask;

    /// <summary>The output size for an input of <paramref name="width"/> × <paramref name="height"/>: the target height with the width following the aspect ratio, else the enlargement.</summary>
    /// <param name="width">Input width in pixels.</param>
    /// <param name="height">Input height in pixels.</param>
    /// <param name="targetHeight">The output height, or 0 to use <paramref name="scale"/>.</param>
    /// <param name="scale">The enlargement of both sides.</param>
    internal static (int Width, int Height) OutputSizeFor(int width, int height, int targetHeight, double scale)
    {
        if (targetHeight > 0)
        {
            return ((int)Math.Round(width * (double)targetHeight / height), targetHeight);
        }

        return ((int)Math.Round(width * scale), (int)Math.Round(height * scale));
    }

    /// <summary>Takes <paramref name="path"/> as the input: an image is decoded, a video probed.</summary>
    /// <param name="path">A video or image file.</param>
    /// <returns>Whether the file is usable (otherwise <see cref="InferencePageViewModel.Notice"/> says why).</returns>
    internal async Task<bool> SetInputAsync(string path)
    {
        if (!File.Exists(path))
        {
            Notice = $"{path} does not exist.";
            return false;
        }

        string name = Path.GetFileName(path);
        try
        {
            if (ImageFiles.IsImage(path))
            {
                var image = await Task.Run(() => ImageFiles.Read(path)).ConfigureAwait(true);
                Accept(new EnhancementInput(path, image, image.Width, image.Height, 1, 0), [image], 0);
                return true;
            }

            bool video = s_videoExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase)
                || _decoders.FileExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase);
            if (!video)
            {
                Notice = $"{name} is neither a video nor an image file.";
                return false;
            }

            if (!_decoders.TryGet(out IVideoDecoder? decoder))
            {
                Notice = VideoSupportNotice;
                return false;
            }

            VideoInfo info = await Task.Run(() => decoder.ProbeAsync(path)).ConfigureAwait(true);
            if (info.Width <= 0 || info.Height <= 0)
            {
                Notice = $"{name} has no video track.";
                return false;
            }

            ImageInput? first = await Task.Run(() => ReadFirstFrameAsync(decoder, path)).ConfigureAwait(true);
            long frames = info.FrameCount > 0 ? info.FrameCount : (long)Math.Round(info.Duration.TotalSeconds * info.FramesPerSecond);
            Accept(new EnhancementInput(path, null, info.Width, info.Height, Math.Max(1, frames), info.FramesPerSecond), first is null ? [] : [first], info.FramesPerSecond);
            return true;
        }
        catch (Exception error) when (error is IOException or InvalidDataException or NotSupportedException or PlatformNotSupportedException or UnauthorizedAccessException)
        {
            Notice = $"{name} cannot be read: {error.Message}";
            return false;
        }
    }

    private void Accept(EnhancementInput input, IReadOnlyList<ImageInput> preview, double framesPerSecond)
    {
        Notice = "";
        Input = input;
        ResultPath = "";
        ResultSummary = "";
        StepText = "";
        Compare.Clear();
        if (preview.Count > 0)
        {
            Player.Load(preview, framesPerSecond > 0 ? framesPerSecond : 24);
        }
        else
        {
            Player.Clear();
        }
    }

    private static async Task<ImageInput?> ReadFirstFrameAsync(IVideoDecoder decoder, string path)
    {
        await foreach (VideoFrame frame in decoder.ReadFramesAsync(path, new VideoDecodeOptions { MaxFrames = 1 }).ConfigureAwait(false))
        {
            return frame.ToImage();
        }

        return null;
    }

    /// <summary>Lets the user pick the video or image to enhance.</summary>
    [RelayCommand]
    private async Task PickInputAsync()
    {
        var media = CanReadVideos
            ? new FileFilter("Videos and images", [.. _decoders.FileExtensions.Concat(ImageFiles.Extensions).Distinct(StringComparer.OrdinalIgnoreCase).Select(extension => $"*{extension}")])
            : ImageFiles.Filter;
        IReadOnlyList<string> files = await Services.Platform.PickFilesAsync("Choose a video or image to enhance", [media, new FileFilter("All files", ["*"])], allowMultiple: false);
        if (files.Count == 1)
        {
            await SetInputAsync(files[0]);
        }
    }

    /// <summary>Forgets the input.</summary>
    [RelayCommand]
    private void ClearInput()
    {
        Input = null;
        Player.Clear();
        Compare.Clear();
        Notice = "";
    }

    /// <inheritdoc/>
    protected override void OnModelChanged(ILoadedModel? model)
    {
        OnPropertyChanged(nameof(RunHint));
        RunCommand.NotifyCanExecuteChanged();
        if (model is { Kind: ModelKinds.VideoEnhancement, State: ResidentState.Ready or ResidentState.Busy } && model.Id != _defaultsOf)
        {
            _defaultsOf = model.Id;
            ModelDefaults = ApplyModelDefaultsAsync(model);
        }
    }

    /// <summary>Sets the enlargement, frames per batch and colour correction to the model's defaults.</summary>
    private async Task ApplyModelDefaultsAsync(ILoadedModel model)
    {
        VideoEnhancementDefaults defaults;
        try
        {
            using var lease = await model.AcquireClientAsync<IVideoEnhancer>(CancellationToken.None).ConfigureAwait(false);
            defaults = lease.Client.Defaults;
        }
        catch (Exception error) when (error is InvalidOperationException or NotSupportedException or ObjectDisposedException)
        {
            return;
        }

        Services.Dispatcher.Post(() =>
        {
            if (SelectedModel?.Id != model.Id)
            {
                return;
            }

            Panel.Apply(new Dictionary<string, string>
            {
                ["scale"] = defaults.Scale.ToString(CultureInfo.InvariantCulture),
                ["batch"] = defaults.FramesPerBatch.ToString(CultureInfo.InvariantCulture),
                ["color"] = defaults.ColorCorrection.ToString(),
            });
        });
    }

    /// <inheritdoc/>
    protected override void OnBusyChanged()
    {
        OnPropertyChanged(nameof(IsEnhancing));
        RunCommand.NotifyCanExecuteChanged();
    }

    private void OnParameterChanged(object? sender, PropertyChangedEventArgs e)
    {
        OnPropertyChanged(nameof(OutputSizeText));
        OnPropertyChanged(nameof(RunHint));
        RunCommand.NotifyCanExecuteChanged();
    }

    private (int Width, int Height)? OutputSize()
        => Input is { } input ? OutputSizeFor(input.Width, input.Height, Panel.GetInt("height"), Panel.GetNumber("scale")) : null;

    /// <summary>Why the page cannot run, or null when it can.</summary>
    private string? InvalidReason()
    {
        if (!HasModel)
        {
            return "Load a video enhancement model first.";
        }

        if (OutputSize() is not { } size)
        {
            return "Choose a video or an image first.";
        }

        if (size.Width < MinOutputSide || size.Height < MinOutputSide)
        {
            return $"The output would be {size.Width} × {size.Height}: choose a larger height or scale.";
        }

        if (size.Width > MaxOutputWidth || size.Height > MaxOutputHeight)
        {
            return $"The output would be {size.Width} × {size.Height}, larger than 8K ({MaxOutputWidth} × {MaxOutputHeight}): choose a smaller height or scale.";
        }

        return null;
    }

    private bool CanRun() => InvalidReason() is null;

    /// <summary>Enhances the input with the chosen model and the current parameters.</summary>
    [RelayCommand(CanExecute = nameof(CanRun))]
    private void Run()
    {
        if (!SeedParser.TryRead(Panel.GetText("seed"), out long? seed))
        {
            Notice = "The seed must be a whole number (leave it empty for a random one).";
            return;
        }

        var model = SelectedModel!;
        var input = Input!;
        var size = OutputSize()!.Value;
        var options = Panel.Options with { Width = size.Width, Height = size.Height, Seed = seed };
        var plan = new EnhancementRun(input, options, Panel.Snapshot());
        string title = $"{Path.GetFileName(input.Path)} → {size.Width} × {size.Height}";
        Panel.Persist();
        Notice = "";
        StepText = "";
        ResultPath = "";
        Queue.Enqueue(title, model.Id, context => EnhanceAsync(model, plan, title, context));
        OnPropertyChanged(nameof(IsEnhancing));
        RunCommand.NotifyCanExecuteChanged();
    }

    /// <summary>Cancels the queued and running enhancements.</summary>
    [RelayCommand]
    private void Cancel() => Queue.CancelAll();

    /// <summary>
    /// One run, streamed end to end: the video is decoded frame by frame into the enhancer, each finished batch is encoded into the output file
    /// and shown in the comparison, so memory stays that of a few batches whatever the clip's length. A cancelled or failed run finishes the
    /// file it has written so far (a valid, shorter video with its sound) and keeps it.
    /// </summary>
    private async Task EnhanceAsync(ILoadedModel model, EnhancementRun plan, string title, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        var token = context.CancellationToken;
        var progress = new EnhancementProgress(plan.Input.Frames, plan.Input.FramesPerSecond);
        var originals = new OriginalFrameQueue();
        await using var output = new EnhancementOutput(plan.Input, _encoders, Services.Pages.GenerationsDirectory("enhanced"));
        VideoEnhancementUpdate? last = null;
        RunOutcome outcome = RunOutcome.Completed;
        string? error = null;
        try
        {
            VideoEnhancementRequest request;
            if (plan.Input.Image is { } image)
            {
                request = VideoEnhancementRequest.FromImage(image);
            }
            else
            {
                if (!_decoders.TryGet(out IVideoDecoder? decoder))
                {
                    throw new NotSupportedException(VideoSupportNotice);
                }

                var frames = decoder.ReadFramesAsync(plan.Input.Path, null, token).Select(frame => frame.ToImage());
                request = VideoEnhancementRequest.FromSource(originals.Tap(frames, token), plan.Input.FramesPerSecond > 0 ? plan.Input.FramesPerSecond : null,
                    plan.Input.Frames);
            }

            Compare.Begin(plan.Input.Frames, plan.Input.FramesPerSecond);
            context.Report(null, "Loading the model");
            using var lease = await model.AcquireClientAsync<IVideoEnhancer>(token).ConfigureAwait(false);
            context.Report(0, "Starting");
            Task shown = Task.CompletedTask;
            await foreach (var update in lease.Client.EnhanceStreamingAsync(request, plan.Options, token).ConfigureAwait(false))
            {
                last = update;
                if (update.Frames is { Count: > 0 } finished)
                {
                    IReadOnlyList<ImageInput> inputs = plan.Input.Image is { } still ? [still] : originals.Take(finished.Count);
                    await output.WriteAsync(finished, update, token).ConfigureAwait(false);
                    // The comparison resizes on the thread pool; waiting for the previous batch keeps at most one batch queued there.
                    await shown.ConfigureAwait(false);
                    shown = Compare.Add(update.FirstFrame, inputs, finished);
                }

                ShowProgress(context, progress.Describe(update, watch.Elapsed));
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
        if (outcome == RunOutcome.Completed && last is not { Phase: VideoEnhancementPhase.Completed, FinishedFrames: > 0 })
        {
            outcome = RunOutcome.Failed;
            error = "The model returned no frames.";
        }

        if (outcome != RunOutcome.Completed)
        {
            string? partial = await FinishPartialAsync(output).ConfigureAwait(false);
            string kept = partial is null ? "" : $" The {output.FramesWritten} frames finished so far were saved as {Path.GetFileName(partial)}.";
            Services.Dispatcher.Post(() =>
            {
                StepText = "";
                ResultPath = partial ?? "";
            });
            if (outcome == RunOutcome.Failed)
            {
                Services.Dispatcher.Post(() => Notice = error! + kept);
                Archived(NewRecord(title, model, plan, outcome, seconds, error, [], Metadata(plan, last, seconds, partial)));
                throw new InvalidOperationException(error);
            }

            if (partial is not null)
            {
                Services.Dispatcher.Post(() => Notice = "Cancelled." + kept);
                Archived(NewRecord(title, model, plan, outcome, seconds, null, [], Metadata(plan, last, seconds, partial)));
            }

            throw new OperationCanceledException(token);
        }

        string? written = await output.CompleteAsync(token).ConfigureAwait(false);
        FinishRun(model, plan, title, output, last!, written, seconds);
    }

    /// <summary>Finishes the file of a run that stopped early, so the frames finished so far stay a valid video; null when there are none.</summary>
    private static async Task<string?> FinishPartialAsync(EnhancementOutput output)
    {
        if (output.FramesWritten == 0)
        {
            return null;
        }

        try
        {
            return await output.CompleteAsync(CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception failure) when (failure is IOException or UnauthorizedAccessException or InvalidOperationException)
        {
            // The file cannot be finished (the disk failed); disposing the output deletes it.
            return null;
        }
    }

    private void ShowProgress(RunContext context, (string Text, double Fraction) progress)
    {
        context.Report(Math.Clamp(progress.Fraction, 0, 1), progress.Text);
        Services.Dispatcher.Post(() => StepText = progress.Text);
    }

    private void FinishRun(ILoadedModel model, EnhancementRun plan, string title, EnhancementOutput output, VideoEnhancementUpdate last, string? written,
        double seconds)
    {
        var metadata = Metadata(plan, last, seconds, written);
        string summary = string.Join(" · ", metadata.Where(pair => pair.Key != OutputMetadataKey).Select(pair => $"{pair.Key}: {pair.Value}"));
        Services.Dispatcher.Post(() =>
        {
            StepText = written is null ? "" : $"Saved {Path.GetFileName(written)}";
            ResultSummary = summary;
            ResultPath = written ?? "";
            Notice = output.WriteError is { } writeError
                ? $"The result could not be written: {writeError}"
                : output.AudioError is { } audioError ? $"The result has no sound: the input's sound track could not be read ({audioError})." : "";
        });

        var files = new Dictionary<string, byte[]> { [ResultStillName] = ImageFiles.ToPng(output.FirstFrame!) };
        Archived(NewRecord(title, model, plan, RunOutcome.Completed, seconds, null, [.. files.Keys], metadata), files);
    }

    private static Dictionary<string, string> Metadata(EnhancementRun plan, VideoEnhancementUpdate? last, double seconds, string? file)
    {
        var metadata = new Dictionary<string, string>
        {
            ["size"] = last is { Size.Width: > 0 } update
                ? $"{plan.Input.Width} × {plan.Input.Height} → {update.Size.Width} × {update.Size.Height}"
                : $"{plan.Input.Width} × {plan.Input.Height}",
            ["frames"] = (last?.FinishedFrames ?? 0).ToString(CultureInfo.InvariantCulture),
            ["seed"] = (last?.Seed ?? 0).ToString(CultureInfo.InvariantCulture),
            ["time"] = $"{seconds:0.0} s",
        };
        if (file is not null)
        {
            metadata[OutputMetadataKey] = file;
        }

        return metadata;
    }

    /// <summary>Opens the result file in the system's viewer.</summary>
    [RelayCommand(CanExecute = nameof(HasResultFile))]
    private void OpenResult() => Services.Platform.Open(ResultPath);

    /// <summary>Asks where to save a copy of the result file.</summary>
    [RelayCommand(CanExecute = nameof(HasResultFile))]
    private async Task SaveResultAsync()
    {
        string extension = Path.GetExtension(ResultPath);
        var filter = new FileFilter(extension.TrimStart('.').ToUpperInvariant(), [$"*{extension}"]);
        if (await Services.Platform.PickSaveFileAsync("Save the enhanced file", Path.GetFileName(ResultPath), filter) is { } path)
        {
            try
            {
                File.Copy(ResultPath, path, overwrite: true);
                StepText = $"Saved {Path.GetFileName(path)}";
            }
            catch (Exception failure) when (failure is IOException or UnauthorizedAccessException)
            {
                Notice = $"The file could not be saved: {failure.Message}";
            }
        }
    }

    /// <inheritdoc/>
    protected override void Restore(RunRecord record)
    {
        Panel.Apply(record.Options);
        Compare.Clear();
        string still = Archive.PathOf(record, ResultStillName);
        if (File.Exists(still))
        {
            Player.Load([ImageFiles.Decode(File.ReadAllBytes(still))], 24);
        }
        else
        {
            Player.Clear();
        }

        ResultPath = record.Metadata.TryGetValue(OutputMetadataKey, out var file) && File.Exists(file) ? file : "";
        StepText = "";
        ResultSummary = string.Join(" · ", record.Metadata.Where(pair => pair.Key != OutputMetadataKey).Select(pair => $"{pair.Key}: {pair.Value}"));
        Notice = record.Outcome == RunOutcome.Failed ? record.Error ?? "" : "";
    }

    private static RunRecord NewRecord(string title, ILoadedModel model, EnhancementRun plan, RunOutcome outcome, double seconds, string? error,
        List<string> fileNames, Dictionary<string, string> metadata) => new()
    {
        Id = PageArchive.NewId(),
        Title = title,
        Model = model.Id,
        Outcome = outcome,
        Seconds = seconds,
        Error = error,
        Input = new Dictionary<string, string>
        {
            ["input"] = plan.Input.Path,
            ["input size"] = $"{plan.Input.Width} × {plan.Input.Height}",
            ["frames"] = plan.Input.Frames.ToString(CultureInfo.InvariantCulture),
        },
        Options = plan.Values,
        Metadata = metadata,
        Files = fileNames,
    };

    private static ParameterPanelViewModel<VideoEnhancementOptions> CreateParameters(StudioServices services)
    {
        ParameterViewModel[] parameters =
        [
            new NumberParameterViewModel("height", "Output height", "Pixels; the width follows the input's aspect ratio. 0 enlarges by the scale instead.", 0, MaxOutputHeight, 8, 0),
            new NumberParameterViewModel("scale", "Scale", "How much both sides grow when no output height is given.", 1, 8, 0.5, 2),
            new NumberParameterViewModel("batch", "Frames per batch", "Frames restored together: fewer need less memory, more keep motion more consistent. 0 = the whole clip at once, long clips in the model's longest batches.", 0, 241, 4, 0),
            new ChoiceParameterViewModel("color", "Colour correction", "How the colours of the result follow the input.", s_colorCorrections),
            new TextParameterViewModel("seed", "Seed", "Empty picks a random seed; a number repeats the result.", "Random"),
        ];
        return new ParameterPanelViewModel<VideoEnhancementOptions>(services.Pages, "enhance", parameters, panel => new VideoEnhancementOptions
        {
            Scale = (float)panel.GetNumber("scale"),
            FramesPerBatch = panel.GetInt("batch"),
            ColorCorrection = Enum.TryParse(panel.GetChoice("color"), out VideoColorCorrection color) ? color : null,
        });
    }
}
