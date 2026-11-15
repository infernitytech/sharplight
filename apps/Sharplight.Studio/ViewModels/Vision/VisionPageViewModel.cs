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
    private const int EmbeddingPreviewValues = 12;

    private static readonly string[] s_kinds =
        [ModelKinds.ImageClassification, ModelKinds.ImageEmbedding, ModelKinds.ObjectDetection, ModelKinds.DepthEstimation];

    private VisionResultFile? _file;
    private ImageInput? _shown;
    private byte[]? _shownPng;
    private byte[]? _storedMain;
    private byte[]? _storedDepth;
    private DepthMap? _depth;
    private int _version;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    public VisionPageViewModel(StudioServices services, INavigator navigator)
        : base(services, navigator, s_kinds, "vision", CreateParameters(services))
    {
        Source = new ImageInputViewModel(services.Platform, text => Notice = text, "Image", "Drop an image here or choose a file (PNG, JPEG, WebP, BMP, GIF).", OnInputChanged);
        foreach (string key in new[] { "threshold", "colormap", "invert" })
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
        _ => "Run",
    };

    /// <summary>What the model does, in words, for the title of the input.</summary>
    public string ModeDescription => Mode switch
    {
        VisionMode.Classify => "Image classification: the model names what the picture shows.",
        VisionMode.Embed => "Image embedding: score candidate labels (models with a text tower, such as CLIP and SigLIP), or get the image's vector.",
        VisionMode.Detect => "Object detection: boxes with labels and scores; the threshold moves after the run.",
        VisionMode.Depth => "Depth estimation: a depth map beside the image.",
        _ => "Load an image model to choose what to do with a picture.",
    };

    /// <summary>The candidate labels box is shown (an embedding model).</summary>
    public bool ShowCandidates => Mode == VisionMode.Embed;

    /// <summary>The result's mode (<c>classification</c>, <c>zero-shot</c>, <c>embedding</c>, <c>detection</c>, <c>depth</c>), or empty.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasResult), nameof(ShowRows), nameof(ShowDepth), nameof(ShowEmbedding), nameof(CanExportImage))]
    public partial string ResultMode { get; set; } = "";

    /// <summary>The image of the result as PNG: the input, or the input with the boxes drawn on it.</summary>
    [ObservableProperty]
    public partial byte[]? MainPng { get; set; }

    /// <summary>The depth map as PNG (depth estimation).</summary>
    [ObservableProperty]
    public partial byte[]? DepthPng { get; set; }

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
    public bool ShowRows => ResultMode is "classification" or "zero-shot" or "detection";

    /// <summary>The result has a depth map.</summary>
    public bool ShowDepth => ResultMode == "depth";

    /// <summary>The result is an embedding vector.</summary>
    public bool ShowEmbedding => ResultMode == "embedding";

    /// <summary>The result has an image to export (annotated boxes or the depth map).</summary>
    public bool CanExportImage => ResultMode is "detection" or "depth";

    /// <summary>The objects shown are fewer than the model found, or there are none (detection).</summary>
    [ObservableProperty]
    public partial string RowsHint { get; set; } = "";

    /// <summary>The latest re-drawing of the annotated image or depth map (for tests).</summary>
    internal Task Refresh { get; private set; } = Task.CompletedTask;

    /// <summary>Why <see cref="RunCommand"/> cannot run, for the hint under the button; empty when it can.</summary>
    public string RunHint => !HasModel ? "Load an image classification, embedding, detection or depth model first." : !Source.HasImage ? "Choose or drop an image first." : "";

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
            _ => VisionMode.None,
        };
        Panel.SetVisible("topk", Mode is VisionMode.Classify or VisionMode.Embed or VisionMode.None);
        Panel.SetVisible("template", Mode == VisionMode.Embed);
        Panel.SetVisible("normalize", Mode == VisionMode.Embed);
        Panel.SetVisible("threshold", Mode == VisionMode.Detect);
        Panel.SetVisible("maxdet", Mode == VisionMode.Detect);
        Panel.SetVisible("colormap", Mode == VisionMode.Depth);
        Panel.SetVisible("invert", Mode == VisionMode.Depth);
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
        _depth = null;
        Rows.Clear();
        MainPng = null;
        DepthPng = null;
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
        string name = Source.Name;
        string path = Source.Path;
        double threshold = Panel.GetNumber("threshold");
        Panel.Persist();
        Notice = "";
        Queue.Enqueue(name, model.Id, context => AnalyzeAsync(model, mode, image, png, name, path, options, values, candidates, threshold, context));
        OnPropertyChanged(nameof(IsRunning));
        RunCommand.NotifyCanExecuteChanged();
    }

    private async Task AnalyzeAsync(ILoadedModel model, VisionMode mode, ImageInput image, byte[]? png, string name, string path, VisionOptions options,
        Dictionary<string, string> values, IReadOnlyList<string> candidates, double threshold, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        var file = new VisionResultFile { Model = model.Id, Image = name, Width = image.Width, Height = image.Height };
        DepthMap? depth = null;
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

            files[ResultName] = JsonSerializer.SerializeToUtf8Bytes(file, StudioJsonContext.Default.VisionResultFile);
            fileNames.Add(ResultName);
            var shownFile = file;
            var shownDepth = depth;
            var shownPng = png;
            Services.Dispatcher.Post(() =>
            {
                Notice = note ?? "";
                Show(shownFile, image, shownPng, shownDepth, null, null, string.Join(" · ", metadata.Select(pair => $"{pair.Key}: {pair.Value}")));
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
                    metadata["top"] = $"{top[0].Label} {top[0].Score:P0}";
                }

                return string.Join(", ", top.Select(label => $"{label.Label} {label.Score:P0}"));
            case "detection":
                int count = file.Boxes!.Count(box => box.Score >= file.Threshold);
                metadata["objects"] = count.ToString(CultureInfo.InvariantCulture);
                return count == 0 ? "No objects above the threshold" : $"{count} object(s): " + string.Join(", ", file.Boxes!.Where(box => box.Score >= file.Threshold).Select(box => box.Label).Distinct().Take(6));
            case "depth":
                return $"Depth range {file.DepthMin:0.##} to {file.DepthMax:0.##}";
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
            annotator.DrawBox(new BoundingBox(box.X1, box.Y1, box.X2, box.Y2), box.ClassIndex, $"{box.Label} {box.Score:P0}");
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

        byte[]? stored = record.Files.Contains(AnnotatedName) && File.Exists(Archive.PathOf(record, AnnotatedName)) ? File.ReadAllBytes(Archive.PathOf(record, AnnotatedName)) : null;
        byte[]? storedDepth = record.Files.Contains(DepthName) && File.Exists(Archive.PathOf(record, DepthName)) ? File.ReadAllBytes(Archive.PathOf(record, DepthName)) : null;
        Notice = "";
        Show(file, image, null, null, stored, storedDepth, string.Join(" · ", record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}")));
        if (image is not null)
        {
            Source.Set(image, source!);
        }
    }

    /// <summary>
    /// Shows a result. A depth map is kept to re-color it; a detection's boxes to move the threshold. For a stored run whose input is
    /// gone only the stored PNGs are shown.
    /// </summary>
    private void Show(VisionResultFile file, ImageInput? image, byte[]? imagePng, DepthMap? depth, byte[]? storedMain, byte[]? storedDepth, string summary)
    {
        _version++;
        _file = file;
        _shown = image;
        _shownPng = imagePng;
        _storedMain = storedMain;
        _storedDepth = storedDepth;
        _depth = depth;
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
        var imagePng = _shownPng;
        Refresh = Task.Run(() =>
        {
            byte[]? main = null;
            byte[]? depthPng = null;
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
        byte[]? bytes = ResultMode == "depth" ? DepthPng : ResultMode == "detection" ? MainPng : null;
        if (bytes is not null)
        {
            string stem = Path.GetFileNameWithoutExtension(_file?.Image ?? "image");
            await Exporter.SaveAsync(Services.Platform, "Export image", $"{stem}-{(ResultMode == "depth" ? "depth" : "annotated")}.png", bytes);
        }
    }

    /// <summary>Copies the result's text (the classes with scores, the objects, or the embedding) to the clipboard.</summary>
    [RelayCommand]
    private Task CopyAsync()
        => Services.Platform.CopyTextAsync(ShowEmbedding ? EmbeddingText : string.Join('\n', Rows.Select(row => $"{row.Label}\t{row.ScoreText}")));

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
            panel.GetBool("invert")), presets);
    }
}
