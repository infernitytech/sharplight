using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Vision;

/// <summary>
/// The Vision page: one page for the image models that read an image, with the mode chosen by the loaded model's kind. Image
/// classification shows the best classes; image embedding scores candidate labels zero-shot (CLIP, SigLIP: models with a text tower)
/// or returns the image's vector; object detection draws boxes with labels and scores above a threshold that can be moved after the
/// run; depth estimation shows the depth map (grayscale or a color map) beside the image. Results export as JSON and annotated
/// or depth PNG. Built from the shared inference blocks.
/// </summary>
internal sealed partial class VisionPageViewModel : InferencePageViewModel
{
    /// <summary>The lowest score a detection run asks for, so the threshold slider can show more objects afterwards without a new run.</summary>
    internal const float DetectionFloor = 0.05f;

    private const string ResultName = "result.json";
    private const string AnnotatedName = "annotated.png";
    private const string DepthName = "depth.png";
    private const string OverlayName = "overlay.png";
    private const string UpscaledName = "upscaled.png";
    private const int EmbeddingPreviewValues = 12;

    private static readonly string[] s_kinds =
    [
        ModelKinds.ImageClassification, ModelKinds.ImageEmbedding, ModelKinds.ObjectDetection, ModelKinds.DepthEstimation,
        ModelKinds.ImageSegmentation, ModelKinds.SemanticSegmentation, ModelKinds.Ocr, ModelKinds.ImageUpscaling,
    ];

    private VisionResultFile? _file;
    private ImageInput? _shown;
    private byte[]? _shownPng;
    private byte[]? _storedMain;
    private byte[]? _storedDepth;
    private DepthMap? _depth;
    private byte[]? _storedAfter;
    private IReadOnlyList<SegmentMask> _masks = [];
    private int _version;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    public VisionPageViewModel(StudioServices services, INavigator navigator)
        : base(services, navigator, s_kinds, "vision", CreateParameters(services))
    {
        Source = new ImageInputViewModel(services.Platform, text => Notice = text, "Image", "Drop an image here or choose a file (PNG, JPEG, WebP, BMP, GIF).", OnInputChanged);
        foreach (string key in new[] { "threshold", "colormap", "invert", "mask" })
        {
            Panel.Parameters.First(parameter => parameter.Key == key).PropertyChanged += (_, _) => RefreshView();
        }

        ApplyMode();
    }

    /// <inheritdoc/>
    public override string Title => "Vision";

    /// <inheritdoc/>
    public override string Icon => "IconScan";

    /// <inheritdoc/>
    public override string KindNoun => "vision";

    /// <summary>The typed parameter panel (maps to <see cref="VisionOptions"/>).</summary>
    public ParameterPanelViewModel<VisionOptions> Panel => (ParameterPanelViewModel<VisionOptions>)Parameters;

    /// <summary>The image to analyze.</summary>
    public ImageInputViewModel Source { get; }

    /// <summary>What the chosen model does (from its kind).</summary>
    public VisionMode Mode { get; private set; }

    /// <summary>Candidate labels for zero-shot scoring, separated by commas or lines.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(RunText))]
    public partial string Candidates { get; set; } = "";

    /// <summary>The page is analyzing (a run is queued or running).</summary>
    public bool IsRunning => Queue.IsBusy;

    /// <summary>The text of the run button.</summary>
    public string RunText => Mode switch
    {
        VisionMode.Classify => "Classify",
        VisionMode.Embed => ParseCandidates(Candidates).Count > 0 ? "Score labels" : "Embed",
        VisionMode.Detect => "Detect objects",
        VisionMode.Depth => "Estimate depth",
        VisionMode.Segment => "Segment",
        VisionMode.Semantic => "Label pixels",
        VisionMode.Ocr => "Read text",
        VisionMode.Upscale => "Upscale",
        _ => "Run",
    };

    /// <summary>What the model does, in words, for the title of the input.</summary>
    public string ModeDescription => Mode switch
    {
        VisionMode.Classify => "Image classification: the model names what the picture shows.",
        VisionMode.Embed => "Image embedding: score candidate labels (models with a text tower, such as CLIP and SigLIP), or get the image's vector.",
        VisionMode.Detect => "Object detection: boxes with labels and scores; the threshold moves after the run.",
        VisionMode.Depth => "Depth estimation: a depth map beside the image.",
        VisionMode.Segment => "Segmentation: run once (it starts from the middle of the image), then click the object (right click: not this) or draw a box around it; the mask follows.",
        VisionMode.Semantic => "Semantic segmentation: every pixel gets a class, shown as a colored overlay with a legend; background-removal models cut the subject out (PNG with transparency).",
        VisionMode.Ocr => "Text recognition: the text the picture shows (a line of text works best).",
        VisionMode.Upscale => "Upscaling: the image enlarged by the model, with a slider to compare before and after; large images are cut into blended tiles.",
        _ => "Load an image model to choose what to do with a picture.",
    };

    /// <summary>The candidate labels box is shown (an embedding model).</summary>
    public bool ShowCandidates => Mode == VisionMode.Embed;

    /// <summary>The result's mode (<c>classification</c>, <c>zero-shot</c>, <c>embedding</c>, <c>detection</c>, <c>depth</c>), or empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasResult), nameof(ShowRows), nameof(ShowDepth), nameof(ShowEmbedding), nameof(CanExportImage), nameof(ShowText), nameof(ShowAfter), nameof(CanPrompt))]
    public partial string ResultMode { get; set; } = "";

    /// <summary>The image of the result as PNG: the input, or the input with the boxes drawn on it.</summary>
    [ObservableProperty]
    public partial byte[]? MainPng { get; set; }

    /// <summary>The depth map as PNG (depth estimation).</summary>
    [ObservableProperty]
    public partial byte[]? DepthPng { get; set; }

    /// <summary>The upscaled image as PNG (upscaling).</summary>
    [ObservableProperty]
    public partial byte[]? AfterPng { get; set; }

    /// <summary>The recognized text (OCR).</summary>
    [ObservableProperty]
    public partial string ResultText { get; set; } = "";

    /// <summary>The classes or objects of the result.</summary>
    public ObservableCollection<VisionRowViewModel> Rows { get; } = [];

    /// <summary>Time and numbers of the last run.</summary>
    [ObservableProperty]
    public partial string ResultSummary { get; set; } = "";

    /// <summary>The embedding's length and first values (embedding).</summary>
    [ObservableProperty]
    public partial string EmbeddingText { get; set; } = "";

    /// <summary>There is a result to show.</summary>
    public bool HasResult => ResultMode.Length > 0;

    /// <summary>The result is a list of classes or objects.</summary>
    public bool ShowRows => ResultMode is "classification" or "zero-shot" or "detection" or "segmentation" or "semantic";

    /// <summary>The result has a depth map.</summary>
    public bool ShowDepth => ResultMode == "depth";

    /// <summary>The result is an embedding vector.</summary>
    public bool ShowEmbedding => ResultMode == "embedding";

    /// <summary>The result has an image to export (annotated boxes or the depth map).</summary>
    public bool CanExportImage => ResultMode is "detection" or "depth" or "segmentation" or "semantic" or "upscale";

    /// <summary>The result is recognized text.</summary>
    public bool ShowText => ResultMode == "ocr";

    /// <summary>The result has an enlarged image beside the original.</summary>
    public bool ShowAfter => ResultMode == "upscale";

    /// <summary>Clicks and drags on the image prompt the segmentation (the model is a promptable segmenter and has encoded the image).</summary>
    public bool CanPrompt => ResultMode == "segmentation";

    /// <summary>The objects shown are fewer than the model found, or there are none (detection).</summary>
    [ObservableProperty]
    public partial string RowsHint { get; set; } = "";

    /// <summary>The latest re-drawing of the annotated image or depth map (for tests).</summary>
    internal Task Refresh { get; private set; } = Task.CompletedTask;

    /// <summary>Why <see cref="RunCommand"/> cannot run, for the hint under the button; empty when it can.</summary>
    public string RunHint => !HasModel ? "Load an image classification, embedding, detection, depth, segmentation, OCR or upscaling model first." : !Source.HasImage ? "Choose or drop an image first." : "";

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

    private void OnInputChanged()
    {
        // A prompt belongs to one image.
        _points.Clear();
        _box = null;
        OnPropertyChanged(nameof(RunHint));
        RunCommand.NotifyCanExecuteChanged();
    }

    private void ApplyMode()
    {
        Mode = SelectedModel?.Kind switch
        {
            ModelKinds.ImageClassification => VisionMode.Classify,
            ModelKinds.ImageEmbedding => VisionMode.Embed,
            ModelKinds.ObjectDetection => VisionMode.Detect,
            ModelKinds.DepthEstimation => VisionMode.Depth,
            ModelKinds.ImageSegmentation => VisionMode.Segment,
            ModelKinds.SemanticSegmentation => VisionMode.Semantic,
            ModelKinds.Ocr => VisionMode.Ocr,
            ModelKinds.ImageUpscaling => VisionMode.Upscale,
            _ => VisionMode.None,
        };
        Panel.SetVisible("topk", Mode is VisionMode.Classify or VisionMode.Embed or VisionMode.None);
        Panel.SetVisible("template", Mode == VisionMode.Embed);
        Panel.SetVisible("normalize", Mode == VisionMode.Embed);
        Panel.SetVisible("threshold", Mode == VisionMode.Detect);
        Panel.SetVisible("maxdet", Mode == VisionMode.Detect);
        Panel.SetVisible("colormap", Mode == VisionMode.Depth);
        Panel.SetVisible("invert", Mode == VisionMode.Depth);
        Panel.SetVisible("segtool", Mode == VisionMode.Segment);
        Panel.SetVisible("mask", Mode == VisionMode.Segment);
        Panel.SetVisible("maxtokens", Mode == VisionMode.Ocr);
        Panel.SetVisible("tile", Mode == VisionMode.Upscale);
        Panel.SetVisible("tilepad", Mode == VisionMode.Upscale);
        OnPropertyChanged(nameof(Mode));
        OnPropertyChanged(nameof(RunText));
        OnPropertyChanged(nameof(ModeDescription));
        OnPropertyChanged(nameof(ShowCandidates));
        OnPropertyChanged(nameof(RunHint));
        RunCommand.NotifyCanExecuteChanged();
    }

    private void ClearResult()
    {
        _version++;
        _file = null;
        _shown = null;
        _shownPng = null;
        _storedMain = null;
        _storedDepth = null;
        _storedAfter = null;
        _depth = null;
        _masks = [];
        _points.Clear();
        _box = null;
        Rows.Clear();
        MainPng = null;
        DepthPng = null;
        AfterPng = null;
        ResultText = "";
        ResultSummary = "";
        EmbeddingText = "";
        RowsHint = "";
        ResultMode = "";
    }

    private bool CanRun() => HasModel && Source.HasImage && Mode != VisionMode.None;

    /// <summary>The labels of a candidate list: separated by commas or lines, trimmed, without duplicates.</summary>
    /// <param name="text">The text the user typed.</param>
    internal static IReadOnlyList<string> ParseCandidates(string text)
        => [.. text.Split([',', '\n', '\r', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Distinct(StringComparer.OrdinalIgnoreCase)];

    /// <summary>Analyzes the image with the chosen model and the current parameters.</summary>
    [RelayCommand(CanExecute = nameof(CanRun))]
    private void Run()
    {
        var model = SelectedModel!;
        var image = Source.Image!;
        var png = Source.Preview;
        var options = Panel.Options;
        var values = Panel.Snapshot();
        var candidates = ParseCandidates(Candidates);
        var mode = Mode;
        var prompt = mode == VisionMode.Segment ? BuildPrompt(image) : null;
        int maskChoice = MaskChoice();
        string name = Source.Name;
        string path = Source.Path;
        double threshold = Panel.GetNumber("threshold");
        Panel.Persist();
        Notice = "";
        Queue.Enqueue(name, model.Id, context => AnalyzeAsync(model, mode, image, png, name, path, options, values, candidates, threshold, prompt, maskChoice, context));
        OnPropertyChanged(nameof(IsRunning));
        RunCommand.NotifyCanExecuteChanged();
    }

    private async Task AnalyzeAsync(ILoadedModel model, VisionMode mode, ImageInput image, byte[]? png, string name, string path, VisionOptions options,
        Dictionary<string, string> values, IReadOnlyList<string> candidates, double threshold, SegmentationPrompt? prompt, int maskChoice, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        var file = new VisionResultFile { Model = model.Id, Image = name, Width = image.Width, Height = image.Height };
        DepthMap? depth = null;
        IReadOnlyList<SegmentMask> masks = [];
        ImageInput? upscaled = null;
        byte[]? semanticPng = null;
        string? note = null;
        RunOutcome outcome = RunOutcome.Completed;
        string? error = null;
        try
        {
            context.Report(null, "Loading the model");
            var token = context.CancellationToken;
            switch (mode)
            {
                case VisionMode.Classify:
                {
                    using var lease = await model.AcquireClientAsync<IImageClassifier>(token).ConfigureAwait(false);
                    context.Report(null, "Classifying");
                    var result = await lease.Client.ClassifyAsync(image, options.Classification, token).ConfigureAwait(false);
                    file.Mode = "classification";
                    file.Labels = [.. result.Predictions.Select(prediction => new VisionLabelRecord(prediction.Index, prediction.Label, prediction.Score))];
                    break;
                }

                case VisionMode.Embed:
                {
                    using var lease = await model.AcquireClientAsync<IImageEmbedder>(token).ConfigureAwait(false);
                    if (candidates.Count > 0 && lease.Client.SupportsText)
                    {
                        context.Report(null, "Scoring the labels");
                        var prompts = candidates.Select(label => options.Template.Contains("{}", StringComparison.Ordinal) ? options.Template.Replace("{}", label, StringComparison.Ordinal) : label).ToList();
                        var result = await lease.Client.ClassifyZeroShotAsync(image, prompts, options.Classification, token).ConfigureAwait(false);
                        file.Mode = "zero-shot";
                        file.Candidates = [.. candidates];
                        file.Labels = [.. result.Predictions.Select(prediction => new VisionLabelRecord(prediction.Index, candidates[prediction.Index], prediction.Score))];
                    }
                    else
                    {
                        if (candidates.Count > 0)
                        {
                            note = "This model has no text tower, so the labels were ignored and the image was embedded.";
                        }

                        context.Report(null, "Embedding");
                        var vector = await lease.Client.EmbedImageAsync(image, options.Embedding, token).ConfigureAwait(false);
                        file.Mode = "embedding";
                        file.Dimensions = vector.Length;
                        file.Embedding = vector;
                    }

                    break;
                }

                case VisionMode.Detect:
                {
                    using var lease = await model.AcquireClientAsync<IObjectDetector>(token).ConfigureAwait(false);
                    context.Report(null, "Detecting");
                    var floor = new ObjectDetectionOptions
                    {
                        ScoreThreshold = Math.Min(DetectionFloor, options.Detection.ScoreThreshold),
                        MaxDetections = options.Detection.MaxDetections,
                        NmsIouThreshold = options.Detection.NmsIouThreshold,
                        ClassAgnosticNms = options.Detection.ClassAgnosticNms,
                    };
                    var result = await lease.Client.DetectAsync(image, floor, token).ConfigureAwait(false);
                    file.Mode = "detection";
                    file.Boxes = [.. result.Objects.Select(item => new VisionBoxRecord(item.ClassIndex, item.Label, item.Score, item.Box.X1, item.Box.Y1, item.Box.X2, item.Box.Y2))];
                    file.Threshold = (float)threshold;
                    break;
                }

                case VisionMode.Depth:
                {
                    using var lease = await model.AcquireClientAsync<IDepthEstimator>(token).ConfigureAwait(false);
                    context.Report(null, "Estimating depth");
                    depth = await lease.Client.EstimateAsync(image, options.Depth, token).ConfigureAwait(false);
                    file.Mode = "depth";
                    file.DepthMin = depth.Min;
                    file.DepthMax = depth.Max;
                    break;
                }

                case VisionMode.Segment:
                {
                    using var lease = await model.AcquireClientAsync<IImageSegmenter>(token).ConfigureAwait(false);
                    var result = await SegmentAsync(lease.Client, model.Id, image, prompt!, options.Segmentation, context, token).ConfigureAwait(false);
                    masks = result.Masks;
                    file.Mode = "segmentation";
                    file.Masks = [.. masks.Select((mask, index) => new VisionMaskRecord(index + 1, mask.Score, mask.Area))];
                    file.ShownMask = ChosenMask(masks, maskChoice) + 1;
                    file.ObjectScore = result.ObjectScore;
                    file.Points = [.. prompt!.Points.Select(point => new VisionPointRecord(point.X, point.Y, point.IsPositive))];
                    file.PromptBox = prompt.Box is { } box ? new VisionBoxRecord(0, "prompt", 1, box.X1, box.Y1, box.X2, box.Y2) : null;
                    break;
                }

                case VisionMode.Semantic:
                {
                    using var lease = await model.AcquireClientAsync<ISemanticSegmenter>(token).ConfigureAwait(false);
                    context.Report(null, "Labelling pixels");
                    var result = await lease.Client.SegmentAsync(image, new SemanticSegmentationOptions(), token).ConfigureAwait(false);
                    file.Mode = "semantic";
                    file.Matte = result.Alpha is not null ? true : null;
                    file.Labels = [.. result.Classes.Select(c => new VisionLabelRecord(c.ClassId, c.Name, c.Pixels / (float)result.Labels.Length))];
                    semanticPng = SemanticOverlay.Render(image, result);
                    break;
                }

                case VisionMode.Ocr:
                {
                    using var lease = await model.AcquireClientAsync<ITextRecognizer>(token).ConfigureAwait(false);
                    context.Report(null, "Reading");
                    var result = await lease.Client.RecognizeAsync(image, options.Recognition, token).ConfigureAwait(false);
                    file.Mode = "ocr";
                    file.Text = result.Text;
                    break;
                }

                case VisionMode.Upscale:
                {
                    using var lease = await model.AcquireClientAsync<IImageUpscaler>(token).ConfigureAwait(false);
                    context.Report(null, "Upscaling");
                    var upscale = new UpscaleOptions
                    {
                        TileSize = options.Upscale.TileSize,
                        TilePadding = options.Upscale.TilePadding,
                        Progress = new Progress<double>(fraction => context.Report(fraction, "Upscaling")),
                    };
                    upscaled = await lease.Client.UpscaleAsync(image, upscale, token).ConfigureAwait(false);
                    file.Mode = "upscale";
                    file.OutputWidth = upscaled.Width;
                    file.OutputHeight = upscaled.Height;
                    break;
                }

                default:
                    throw new NotSupportedException("The chosen model is not a vision model.");
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
        var metadata = new Dictionary<string, string>
        {
            ["image"] = $"{image.Width} × {image.Height}",
            ["time"] = $"{seconds:0.0} s",
        };
        var files = new Dictionary<string, byte[]>();
        var fileNames = new List<string>();
        string preview = "";
        if (outcome == RunOutcome.Completed)
        {
            png ??= ImageFiles.ToPng(image);
            preview = Describe(file, metadata);
            if (file.Mode == "detection")
            {
                files[AnnotatedName] = ImageFiles.ToPng(Annotate(image, file.Boxes!, (float)threshold));
                fileNames.Add(AnnotatedName);
            }
            else if (file.Mode == "depth")
            {
                files[DepthName] = ImageFiles.ToPng(DepthColormap.Render(depth!, options.Colormap, options.Invert));
                fileNames.Add(DepthName);
            }
            else if (file.Mode == "segmentation")
            {
                files[OverlayName] = ImageFiles.ToPng(SegmentationOverlay.Render(image, masks.Count > 0 ? masks[ChosenMask(masks, maskChoice)] : null, prompt!.Points, prompt.Box));
                fileNames.Add(OverlayName);
            }
            else if (file.Mode == "semantic")
            {
                files[OverlayName] = semanticPng!;
                fileNames.Add(OverlayName);
            }
            else if (file.Mode == "upscale")
            {
                files[UpscaledName] = ImageFiles.ToPng(upscaled!);
                fileNames.Add(UpscaledName);
            }

            files[ResultName] = JsonSerializer.SerializeToUtf8Bytes(file, StudioJsonContext.Default.VisionResultFile);
            fileNames.Add(ResultName);
            var shownFile = file;
            var shownDepth = depth;
            var shownPng = png;
            var shownMasks = masks;
            byte[]? shownAfter = file.Mode == "upscale" ? files[UpscaledName] : null;
            Services.Dispatcher.Post(() =>
            {
                Notice = note ?? "";
                _masks = shownMasks;
                Show(shownFile, image, shownPng, shownDepth, semanticPng, null, string.Join(" · ", metadata.Select(pair => $"{pair.Key}: {pair.Value}")), shownAfter);
            });
        }

        if (outcome != RunOutcome.Cancelled)
        {
            var input = new Dictionary<string, string> { ["file"] = name, ["path"] = path };
            if (candidates.Count > 0 && mode == VisionMode.Embed)
            {
                input["labels"] = string.Join(", ", candidates);
            }

            var record = new RunRecord
            {
                Id = PageArchive.NewId(),
                Title = name,
                Model = model.Id,
                Outcome = outcome,
                Seconds = seconds,
                Error = error,
                Input = input,
                Options = values,
                Metadata = metadata,
                Preview = preview,
                Files = fileNames,
            };
            Archived(record, outcome == RunOutcome.Completed ? files : null);
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

    /// <summary>A short text of the result for the history and the gallery card; adds the facts to <paramref name="metadata"/>.</summary>
    private static string Describe(VisionResultFile file, Dictionary<string, string> metadata)
    {
        metadata["mode"] = file.Mode;
        switch (file.Mode)
        {
            case "classification" or "zero-shot":
                var top = file.Labels!.Take(3).ToList();
                if (top.Count > 0)
                {
                    metadata["top"] = $"{top[0].Label} {VisionRowViewModel.Percent(top[0].Score)}";
                }

                return string.Join(", ", top.Select(label => $"{label.Label} {VisionRowViewModel.Percent(label.Score)}"));
            case "detection":
                int count = file.Boxes!.Count(box => box.Score >= file.Threshold);
                metadata["objects"] = count.ToString(CultureInfo.InvariantCulture);
                return count == 0 ? "No objects above the threshold" : $"{count} object(s): " + string.Join(", ", file.Boxes!.Where(box => box.Score >= file.Threshold).Select(box => box.Label).Distinct().Take(6));
            case "depth":
                return $"Depth range {file.DepthMin:0.##} to {file.DepthMax:0.##}";
            case "semantic":
                metadata["classes"] = file.Labels!.Count.ToString(CultureInfo.InvariantCulture);
                return file.Matte == true ? "Background removed (alpha matte)" : file.Labels!.Count == 0 ? "No pixels labelled" : string.Join(", ", file.Labels!.Take(4).Select(label => $"{label.Label} {VisionRowViewModel.Percent(label.Score)}"));
            case "segmentation":
                var best = file.Masks!.MaxBy(mask => mask.Score);
                metadata["masks"] = file.Masks!.Count.ToString(CultureInfo.InvariantCulture);
                return best is null ? "No mask" : $"{file.Masks!.Count} candidate mask(s), best {VisionRowViewModel.Percent(best.Score)} ({best.Area} px)";
            case "ocr":
                metadata["characters"] = file.Text!.Length.ToString(CultureInfo.InvariantCulture);
                return file.Text.Length == 0 ? "No text found" : file.Text.Length > 120 ? file.Text[..120] + "…" : file.Text;
            case "upscale":
                metadata["output"] = $"{file.OutputWidth} × {file.OutputHeight}";
                return $"{file.Width} × {file.Height} to {file.OutputWidth} × {file.OutputHeight}";
            default:
                metadata["dimensions"] = file.Dimensions.GetValueOrDefault().ToString(CultureInfo.InvariantCulture);
                return $"{file.Dimensions} dimensions";
        }
    }

    /// <summary>Draws the boxes at or above the threshold on a copy of the image.</summary>
    private static ImageInput Annotate(ImageInput image, IEnumerable<VisionBoxRecord> boxes, float threshold)
    {
        var annotator = new ImageAnnotator(image);
        foreach (var box in boxes.Where(box => box.Score >= threshold))
        {
            annotator.DrawBox(new BoundingBox(box.X1, box.Y1, box.X2, box.Y2), box.ClassIndex, $"{box.Label} {VisionRowViewModel.Percent(box.Score)}");
        }

        return annotator.ToImage();
    }

    /// <inheritdoc/>
    protected override void Restore(RunRecord record)
    {
        string path = Archive.PathOf(record, ResultName);
        if (!File.Exists(path))
        {
            Notice = "The stored result is gone.";
            return;
        }

        var file = PageJson.Read(path, StudioJsonContext.Default.VisionResultFile);
        Panel.Apply(record.Options);
        Candidates = file.Candidates is { Count: > 0 } ? string.Join(", ", file.Candidates) : "";
        ImageInput? image = null;
        if (record.Input.TryGetValue("path", out var source) && File.Exists(source))
        {
            try
            {
                image = ImageFiles.Read(source);
            }
            catch (Exception error) when (error is IOException or InvalidDataException or NotSupportedException)
            {
                // The original is gone or unreadable: the stored images are shown instead.
            }
        }

        string mainName = file.Mode is "segmentation" or "semantic" ? OverlayName : AnnotatedName;
        byte[]? stored = record.Files.Contains(mainName) && File.Exists(Archive.PathOf(record, mainName)) ? File.ReadAllBytes(Archive.PathOf(record, mainName)) : null;
        byte[]? storedAfter = record.Files.Contains(UpscaledName) && File.Exists(Archive.PathOf(record, UpscaledName)) ? File.ReadAllBytes(Archive.PathOf(record, UpscaledName)) : null;
        byte[]? storedDepth = record.Files.Contains(DepthName) && File.Exists(Archive.PathOf(record, DepthName)) ? File.ReadAllBytes(Archive.PathOf(record, DepthName)) : null;
        Notice = "";
        _masks = [];
        Show(file, image, null, null, stored, storedDepth, string.Join(" · ", record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}")), storedAfter);
        if (image is not null)
        {
            Source.Set(image, source!);
        }
    }

    /// <summary>
    /// Shows a result. A depth map is kept to re-color it; a detection's boxes to move the threshold. For a stored run whose input is
    /// gone only the stored PNGs are shown.
    /// </summary>
    private void Show(VisionResultFile file, ImageInput? image, byte[]? imagePng, DepthMap? depth, byte[]? storedMain, byte[]? storedDepth, string summary, byte[]? storedAfter = null)
    {
        _version++;
        _file = file;
        _shown = image;
        _shownPng = imagePng;
        _storedMain = storedMain;
        _storedDepth = storedDepth;
        _storedAfter = storedAfter;
        _depth = depth;
        ResultText = file.Text ?? "";
        if (file.Mode == "segmentation")
        {
            _points.Clear();
            _points.AddRange((file.Points ?? []).Select(point => new SegmentationPoint(point.X, point.Y, point.Positive)));
            _box = file.PromptBox is { } box ? new BoundingBox(box.X1, box.Y1, box.X2, box.Y2) : null;
        }

        ResultSummary = summary;
        EmbeddingText = file.Embedding is { } vector
            ? $"{vector.Length} dimensions. First values: " + string.Join(", ", vector.Take(EmbeddingPreviewValues).Select(value => value.ToString("0.000", CultureInfo.InvariantCulture))) + (vector.Length > EmbeddingPreviewValues ? ", …" : "")
            : "";
        ResultMode = file.Mode;
        RefreshView();
    }

    /// <summary>Redraws what depends on the threshold and the color map: the rows, the annotated image, the depth image.</summary>
    private void RefreshView()
    {
        if (_file is not { } file)
        {
            return;
        }

        int version = ++_version;
        float threshold = (float)Panel.GetNumber("threshold");
        string colormap = Panel.GetChoice("colormap");
        bool invert = Panel.GetBool("invert");
        var image = _shown;
        var depth = _depth;
        var storedMain = _storedMain;
        var storedDepth = _storedDepth;
        var storedAfter = _storedAfter;
        var imagePng = _shownPng;
        var masks = _masks;
        var points = _points.ToList();
        var box = _box;
        int maskChoice = MaskChoice();
        Refresh = Task.Run(() =>
        {
            byte[]? main = null;
            byte[]? depthPng = null;
            byte[]? afterPng = null;
            var rows = new List<VisionRowViewModel>();
            string hint = "";
            switch (file.Mode)
            {
                case "detection":
                    var kept = file.Boxes!.Where(box => box.Score >= threshold).ToList();
                    rows = [.. kept.Select(box => new VisionRowViewModel(box.Label, box.Score, $"{box.X1:0},{box.Y1:0} to {box.X2:0},{box.Y2:0}", box.ClassIndex))];
                    hint = kept.Count == file.Boxes!.Count ? "" : $"{file.Boxes.Count - kept.Count} more below the threshold.";
                    if (kept.Count == 0)
                    {
                        hint = "No objects above the threshold; lower it to see more.";
                    }

                    main = image is not null ? ImageFiles.ToPng(Annotate(image, kept, 0)) : storedMain;
                    break;
                case "depth":
                    main = image is not null ? imagePng ??= ImageFiles.ToPng(image) : null;
                    depthPng = depth is not null ? ImageFiles.ToPng(DepthColormap.Render(depth, colormap, invert)) : storedDepth;
                    break;
                case "segmentation":
                    rows = [.. (file.Masks ?? []).Select(mask => new VisionRowViewModel($"Mask {mask.Index}", mask.Score, $"{mask.Area} px"))];
                    hint = rows.Count == 0 ? "The model found no mask." : "Click the object, or draw a box around it, to prompt again.";
                    main = image is not null && masks.Count > 0
                        ? ImageFiles.ToPng(SegmentationOverlay.Render(image, masks[ChosenMask(masks, maskChoice)], points, box))
                        : image is not null ? ImageFiles.ToPng(SegmentationOverlay.Render(image, null, points, box)) : storedMain;
                    break;
                case "semantic":
                    rows = [.. file.Labels!.Select(label => new VisionRowViewModel(label.Label, label.Score, "", -1, file.Matte == true ? null : SemanticOverlay.ColorOf(label.Index)))];
                    hint = file.Matte == true ? "The subject is cut out; export the PNG for the transparent image." : "Share of the image per class.";
                    main = storedMain;
                    break;
                case "upscale":
                    main = image is not null ? imagePng ??= ImageFiles.ToPng(image) : null;
                    afterPng = storedAfter;
                    break;
                case "classification" or "zero-shot":
                    rows = [.. file.Labels!.Select(label => new VisionRowViewModel(label.Label, label.Score))];
                    main = image is not null ? imagePng ??= ImageFiles.ToPng(image) : null;
                    break;
                default:
                    main = image is not null ? imagePng ??= ImageFiles.ToPng(image) : null;
                    break;
            }

            Services.Dispatcher.Post(() =>
            {
                if (version != _version)
                {
                    return;
                }

                Rows.Clear();
                foreach (var row in rows)
                {
                    Rows.Add(row);
                }

                RowsHint = hint;
                MainPng = main;
                DepthPng = depthPng;
                AfterPng = afterPng;
            });
        });
    }

    /// <summary>The result as it is exported: a detection keeps the objects above the current threshold.</summary>
    private VisionResultFile? CurrentResult()
    {
        if (_file is not { } file)
        {
            return null;
        }

        if (file.Mode != "detection")
        {
            return file;
        }

        float threshold = (float)Panel.GetNumber("threshold");
        return new VisionResultFile
        {
            Mode = file.Mode, Model = file.Model, Image = file.Image, Width = file.Width, Height = file.Height,
            Boxes = [.. file.Boxes!.Where(box => box.Score >= threshold)], Threshold = threshold,
        };
    }

    /// <summary>Saves the result as JSON where the user chooses.</summary>
    [RelayCommand]
    private async Task ExportJsonAsync()
    {
        if (CurrentResult() is { } result)
        {
            string stem = Path.GetFileNameWithoutExtension(Source.Name.Length > 0 ? Source.Name : result.Image);
            await Exporter.SaveAsync(Services.Platform, "Export result", $"{(stem.Length == 0 ? "result" : stem)}.json", JsonSerializer.SerializeToUtf8Bytes(result, StudioJsonContext.Default.VisionResultFile));
        }
    }

    /// <summary>Saves the annotated image (detection) or the depth map (depth estimation) as PNG where the user chooses.</summary>
    [RelayCommand]
    private async Task ExportImageAsync()
    {
        byte[]? bytes = ResultMode switch
        {
            "depth" => DepthPng,
            "detection" or "segmentation" or "semantic" => MainPng,
            "upscale" => AfterPng,
            _ => null,
        };
        if (bytes is not null)
        {
            string stem = Path.GetFileNameWithoutExtension(_file?.Image ?? "image");
            string suffix = ResultMode switch { "depth" => "depth", "segmentation" => "mask", "semantic" => "segmented", "upscale" => "upscaled", _ => "annotated" };
            await Exporter.SaveAsync(Services.Platform, "Export image", $"{stem}-{suffix}.png", bytes);
        }
    }

    /// <summary>Copies the result's text (the classes with scores, the objects, or the embedding) to the clipboard.</summary>
    [RelayCommand]
    private Task CopyAsync()
        => Services.Platform.CopyTextAsync(ShowText ? ResultText : ShowEmbedding ? EmbeddingText : string.Join('\n', Rows.Select(row => $"{row.Label}\t{row.ScoreText}")));

    /// <summary>Cancels the queued and running analyses.</summary>
    [RelayCommand]
    private void Cancel() => Queue.CancelAll();

    private static ParameterPanelViewModel<VisionOptions> CreateParameters(StudioServices services)
    {
        ParameterViewModel[] parameters =
        [
            new NumberParameterViewModel("topk", "Top classes", "How many of the best classes to list.", 1, 20, 1, 5),
            new TextParameterViewModel("template", "Label template", "Zero-shot: the text a label is put into; {} stands for the label.", "a photo of {}", "a photo of {}"),
            new BoolParameterViewModel("normalize", "Normalize the embedding", "Scale the vector to unit length (what similarity search uses).", defaultValue: true),
            new NumberParameterViewModel("threshold", "Score threshold", "Objects below this score are hidden; moving it needs no new run.", 0.05, 1, 0.05, 0.5),
            new NumberParameterViewModel("maxdet", "Most objects", "At most this many objects (the best ones).", 1, 300, 1, 100),
            new ChoiceParameterViewModel("colormap", "Depth colors", "How the depth map is colored (near is bright in grayscale).",
                [new(DepthColormap.Gray, "Grayscale"), new(DepthColormap.Turbo, "Turbo"), new(DepthColormap.Viridis, "Viridis"), new(DepthColormap.Inferno, "Inferno")], DepthColormap.Turbo),
            new BoolParameterViewModel("invert", "Invert the depth colors", "Swap near and far."),
            new ChoiceParameterViewModel("segtool", "Prompt tool", "What dragging on the segmented image does: click points (right click: not the object) or draw a box.",
                [new(SegmentTool.Point, "Points"), new(SegmentTool.Box, "Box")], SegmentTool.Point),
            new ChoiceParameterViewModel("mask", "Mask shown", "Which candidate mask the overlay shows: the one with the best predicted IoU, or a numbered one.",
                [new("best", "Best"), new("1", "Mask 1"), new("2", "Mask 2"), new("3", "Mask 3")], "best"),
            new NumberParameterViewModel("maxtokens", "Most tokens", "The longest text the recognizer writes, in tokens.", 8, 512, 8, 128),
            new NumberParameterViewModel("tile", "Tile size", "Large images are cut into tiles of this many input pixels (0: the whole image at once, which needs the most memory).", 0, 1024, 32, 192),
            new NumberParameterViewModel("tilepad", "Tile padding", "Context in pixels that neighbouring tiles share; the overlap is blended, so more padding hides seams better.", 0, 64, 4, 16),
        ];
        ParameterPreset[] presets =
        [
            new("Default", new() { ["topk"] = "5", ["threshold"] = "0.5" }, BuiltIn: true),
            new("Strict", new() { ["topk"] = "3", ["threshold"] = "0.75" }, BuiltIn: true),
            new("Sensitive", new() { ["topk"] = "10", ["threshold"] = "0.25" }, BuiltIn: true),
        ];
        return new ParameterPanelViewModel<VisionOptions>(services.Pages, "vision", parameters, panel => new VisionOptions(
            new ImageClassificationOptions { TopK = panel.GetInt("topk") },
            new ObjectDetectionOptions { ScoreThreshold = (float)panel.GetNumber("threshold"), MaxDetections = panel.GetInt("maxdet") },
            new DepthEstimationOptions { ResizeToInput = true },
            new ImageEmbeddingOptions { Normalize = panel.GetBool("normalize") },
            panel.GetText("template"),
            panel.GetChoice("colormap"),
            panel.GetBool("invert"),
            new SegmentationOptions { MultipleMasks = true },
            new TextRecognitionOptions { MaxTokens = panel.GetInt("maxtokens") },
            new UpscaleOptions { TileSize = panel.GetInt("tile"), TilePadding = panel.GetInt("tilepad") }), presets);
    }
}
