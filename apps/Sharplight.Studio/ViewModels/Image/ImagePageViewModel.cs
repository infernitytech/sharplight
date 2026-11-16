using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Imaging;
using Sharplight.Studio.Services.Pages;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Vision;

namespace Sharplight.Studio.ViewModels.Image;

/// <summary>
/// The Image page (kind <c>image-generation</c>): a prompt (and negative prompt) goes to the chosen model's
/// <see cref="IImageGenerator"/>; the streaming call reports every denoising step and a live preview. An input image makes the run
/// image to image (with a strength), an input image and a mask make it inpainting. The results are PNG files carrying the prompt and
/// parameters as text metadata; the seed can be reused. Built from the shared inference blocks.
/// </summary>
internal sealed partial class ImagePageViewModel : InferencePageViewModel
{
    private const string ParametersName = "parameters.json";
    private const int PromptTitleLength = 70;

    private static readonly ParameterChoice[] s_sizes =
    [
        new("native", "Model default"), new("128x128", "128 × 128"), new("256x256", "256 × 256"), new("384x384", "384 × 384"), new("512x512", "512 × 512"),
        new("512x768", "512 × 768 (portrait)"), new("768x512", "768 × 512 (landscape)"), new("768x768", "768 × 768"), new("1024x1024", "1024 × 1024"),
    ];

    private static readonly ParameterChoice[] s_samplers =
    [
        new("", "Model default"), new(Samplers.Euler, "Euler"), new(Samplers.EulerKarras, "Euler (Karras)"), new(Samplers.EulerAncestral, "Euler ancestral"),
        new(Samplers.Ddim, "DDIM"), new(Samplers.Pndm, "PNDM"), new(Samplers.DpmPlusPlus2M, "DPM++ 2M"), new(Samplers.DpmPlusPlus2MKarras, "DPM++ 2M (Karras)"),
    ];

    private const int GenericSteps = 20;
    private const double GenericGuidance = 7.5;

    private ImageParametersFile? _last;

    /// <summary>The model whose distilled defaults (few steps, no guidance) the panel was set to, or null.</summary>
    private string? _distilledDefaultsOf;

    /// <param name="services">The services.</param>
    /// <param name="navigator">Moves to the library and Discover.</param>
    public ImagePageViewModel(StudioServices services, INavigator navigator)
        : base(services, navigator, ModelKinds.ImageGeneration, "image", CreateParameters(services))
    {
        Source = new ImageInputViewModel(services.Platform, text => Notice = text, "Input image", "Optional: start from an image (image to image). Drop a file here or choose one.", OnSourceChanged);
        Mask = new ImageInputViewModel(services.Platform, text => Notice = text, "Mask", "Optional: with an input image, bright areas are regenerated, the rest stays (inpainting).", OnSourceChanged);
        OnSourceChanged();
    }

    /// <inheritdoc/>
    public override string Title => "Image";

    /// <inheritdoc/>
    public override string Icon => "IconImage";

    /// <inheritdoc/>
    public override string KindNoun => "image generation";

    /// <summary>The typed parameter panel (maps to <see cref="ImageGenerationOptions"/>).</summary>
    public ParameterPanelViewModel<ImageGenerationOptions> Panel => (ParameterPanelViewModel<ImageGenerationOptions>)Parameters;

    /// <summary>The input image of an image-to-image run.</summary>
    public ImageInputViewModel Source { get; }

    /// <summary>The mask of an inpainting run (used with <see cref="Source"/>).</summary>
    public ImageInputViewModel Mask { get; }

    /// <summary>What the image should show.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(RunCommand))]
    [NotifyPropertyChangedFor(nameof(RunHint))]
    public partial string Prompt { get; set; } = "";

    /// <summary>What the image should not show.</summary>
    [ObservableProperty]
    public partial string NegativePrompt { get; set; } = "";

    /// <summary>The page is generating (a run is queued or running).</summary>
    public bool IsGenerating => Queue.IsBusy;

    /// <summary>The images of the last result.</summary>
    public ObservableCollection<GeneratedImageViewModel> Images { get; } = [];

    /// <summary>The image shown large.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasResult), nameof(ShowPreview), nameof(ShowPlaceholder), nameof(CanReuseSeed), nameof(SelectedPng))]
    public partial GeneratedImageViewModel? SelectedImage { get; set; }

    /// <summary>The shown image's PNG, or null.</summary>
    public byte[]? SelectedPng => SelectedImage?.Png;

    /// <summary>The live preview of the image being generated (at latent resolution), as PNG, or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowPreview), nameof(ShowPlaceholder))]
    public partial byte[]? PreviewPng { get; set; }

    /// <summary>The step the run is at, or empty.</summary>
    [ObservableProperty]
    public partial string StepText { get; set; } = "";

    /// <summary>The settings and the time of the last result.</summary>
    [ObservableProperty]
    public partial string ResultSummary { get; set; } = "";

    /// <summary>There is a result to show.</summary>
    public bool HasResult => SelectedImage is not null;

    /// <summary>The preview of a running generation is shown.</summary>
    public bool ShowPreview => PreviewPng is not null && IsGenerating;

    /// <summary>Nothing is shown yet.</summary>
    public bool ShowPlaceholder => !HasResult && !ShowPreview;

    /// <summary>The result has several images, so the thumbnails are listed.</summary>
    public bool HasSeveral => Images.Count > 1;

    /// <summary>The selected image's seed can be taken over.</summary>
    public bool CanReuseSeed => SelectedImage is not null;

    /// <summary>The run is image to image or inpainting (an input image is chosen).</summary>
    public bool UsesInputImage => Source.HasImage;

    /// <summary>Why <see cref="RunCommand"/> cannot run, for the hint under the button; empty when it can.</summary>
    public string RunHint => !HasModel ? "Load an image generation model first." : Prompt.Trim().Length == 0 ? "Type a prompt first." : "";

    /// <inheritdoc/>
    protected override void OnModelChanged(ILoadedModel? model)
    {
        OnPropertyChanged(nameof(RunHint));
        RunCommand.NotifyCanExecuteChanged();
        if (model is { Kind: ModelKinds.ImageGeneration, State: ResidentState.Ready or ResidentState.Busy })
        {
            ModelDefaults = ApplyModelDefaultsAsync(model);
        }
    }

    /// <summary>The latest reading of a model's defaults (for tests).</summary>
    internal Task ModelDefaults { get; private set; } = Task.CompletedTask;

    /// <summary>
    /// Sets steps, guidance and sampler to a distilled model's defaults (SDXL-Turbo, FLUX.1-schnell: 4 steps without guidance; FLUX.1-dev:
    /// 28 steps with its embedded guidance 3.5), since the usual 20 steps
    /// with guidance 7.5 spoil its pictures; switching back to a regular model restores the usual values. Regular models keep what the
    /// user chose.
    /// </summary>
    private async Task ApplyModelDefaultsAsync(ILoadedModel model)
    {
        ImageGenerationDefaults? defaults;
        try
        {
            using var lease = await model.AcquireClientAsync<IImageGenerator>(CancellationToken.None).ConfigureAwait(false);
            defaults = lease.Client.GetService(typeof(ImageGenerationDefaults)) as ImageGenerationDefaults;
        }
        catch (Exception error) when (error is InvalidOperationException or NotSupportedException or ObjectDisposedException)
        {
            return;
        }

        if (defaults is null)
        {
            return;
        }

        bool distilled = defaults.GuidanceScale <= 1f || !defaults.ClassifierFreeGuidance;
        Services.Dispatcher.Post(() =>
        {
            if (SelectedModel?.Id != model.Id)
            {
                return;
            }

            if (distilled)
            {
                Panel.Apply(new Dictionary<string, string>
                {
                    ["steps"] = defaults.Steps.ToString(CultureInfo.InvariantCulture),
                    ["guidance"] = defaults.GuidanceScale.ToString(CultureInfo.InvariantCulture),
                    ["sampler"] = "",
                });
                _distilledDefaultsOf = model.Id;
                Notice = defaults.GuidanceScale <= 1f
                    ? $"{model.Id} is a distilled model: set to its defaults, {defaults.Steps} steps without guidance."
                    : $"{model.Id} is a guidance-distilled model: set to its defaults, {defaults.Steps} steps, guidance {defaults.GuidanceScale.ToString(CultureInfo.InvariantCulture)}.";
            }
            else if (_distilledDefaultsOf is not null)
            {
                Panel.Apply(new Dictionary<string, string>
                {
                    ["steps"] = GenericSteps.ToString(CultureInfo.InvariantCulture),
                    ["guidance"] = GenericGuidance.ToString(CultureInfo.InvariantCulture),
                });
                _distilledDefaultsOf = null;
                Notice = "";
            }
        });
    }

    /// <inheritdoc/>
    protected override void OnBusyChanged()
    {
        OnPropertyChanged(nameof(IsGenerating));
        OnPropertyChanged(nameof(ShowPreview));
        OnPropertyChanged(nameof(ShowPlaceholder));
        RunCommand.NotifyCanExecuteChanged();
    }

    /// <inheritdoc/>
    protected override void Restore(RunRecord record)
    {
        string parametersPath = Archive.PathOf(record, ParametersName);
        if (!File.Exists(parametersPath))
        {
            Notice = "The stored images are gone.";
            return;
        }

        var file = PageJson.Read(parametersPath, StudioJsonContext.Default.ImageParametersFile);
        var shown = new List<GeneratedImageViewModel>();
        foreach (string name in record.Files.Where(name => name.StartsWith("image-", StringComparison.Ordinal) && name.EndsWith(".png", StringComparison.Ordinal)))
        {
            string path = Archive.PathOf(record, name);
            if (File.Exists(path))
            {
                byte[] png = File.ReadAllBytes(path);
                shown.Add(new GeneratedImageViewModel(shown.Count + 1, png, ImageFiles.Decode(png), file.Seed + shown.Count));
            }
        }

        if (shown.Count == 0)
        {
            Notice = "The stored images are gone.";
            return;
        }

        Notice = "";
        Panel.Apply(file.Options.Count > 0 ? file.Options : record.Options);
        Prompt = file.Prompt;
        NegativePrompt = file.NegativePrompt;
        _last = file;
        Show(shown, string.Join(" · ", record.Metadata.Select(pair => $"{pair.Key}: {pair.Value}")));
    }

    private void OnSourceChanged()
    {
        Panel.SetVisible("strength", Source.HasImage);
        OnPropertyChanged(nameof(UsesInputImage));
    }

    private bool CanRun() => HasModel && Prompt.Trim().Length > 0;

    /// <summary>Generates the images with the chosen model and the current parameters.</summary>
    [RelayCommand(CanExecute = nameof(CanRun))]
    private void Run()
    {
        if (!SeedParser.TryRead(Panel.GetText("seed"), out long? seed))
        {
            Notice = "The seed must be a whole number (leave it empty for a random one).";
            return;
        }

        var model = SelectedModel!;
        var options = Panel.Options;
        var values = Panel.Snapshot();
        var plan = new GenerationPlan(Prompt.Trim(), NegativePrompt.Trim(), options, seed, Source.Image, Source.Image is not null ? Mask.Image : null,
            (float)Panel.GetNumber("strength"), values, Panel.GetBool("preview"));
        string title = plan.Prompt.Length > PromptTitleLength ? plan.Prompt[..PromptTitleLength] + "…" : plan.Prompt;
        Panel.Persist();
        Notice = "";
        StepText = "";
        PreviewPng = null;
        Queue.Enqueue(title, model.Id, context => GenerateAsync(model, plan, title, context));
        OnPropertyChanged(nameof(IsGenerating));
        RunCommand.NotifyCanExecuteChanged();
    }

    private async Task GenerateAsync(ILoadedModel model, GenerationPlan plan, string title, RunContext context)
    {
        var watch = Stopwatch.StartNew();
        ImageGenerationResult? result = null;
        RunOutcome outcome = RunOutcome.Completed;
        string? error = null;
        try
        {
            context.Report(null, "Loading the model");
            using var lease = await model.AcquireClientAsync<IImageGenerator>(context.CancellationToken).ConfigureAwait(false);
            var request = new ImageGenerationRequest
            {
                Prompt = plan.Prompt,
                NegativePrompt = plan.NegativePrompt.Length > 0 ? plan.NegativePrompt : null,
                Image = plan.Image,
                Strength = plan.Strength,
                Mask = plan.Mask,
            };
            var options = plan.Options with { Seed = plan.Seed, PreviewInterval = plan.Preview ? Math.Max(1, (plan.Options.Steps ?? 20) / 10) : 0 };
            context.Report(0, "Starting");
            await foreach (var update in lease.Client.GenerateStreamingAsync(request, options, context.CancellationToken).ConfigureAwait(false))
            {
                string step = $"Step {update.Step} of {update.TotalSteps}";
                context.Report(update.TotalSteps > 0 ? (double)update.Step / update.TotalSteps : null, step);
                if (update.Preview is { } preview)
                {
                    byte[] png = ImageFiles.ToPng(preview);
                    Services.Dispatcher.Post(() =>
                    {
                        PreviewPng = png;
                        StepText = step;
                    });
                }
                else
                {
                    Services.Dispatcher.Post(() => StepText = step);
                }

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
            error = "The model returned no image.";
        }

        if (outcome != RunOutcome.Completed)
        {
            Services.Dispatcher.Post(() =>
            {
                PreviewPng = null;
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

        var finished = result!;
        var first = finished.Images[0];
        var file = new ImageParametersFile
        {
            Model = model.Id,
            Prompt = plan.Prompt,
            NegativePrompt = plan.NegativePrompt,
            Seed = finished.Seed,
            Steps = plan.Options.Steps ?? 0,
            Guidance = plan.Options.GuidanceScale ?? 0,
            Sampler = plan.Options.Scheduler ?? "",
            Width = first.Width,
            Height = first.Height,
            Strength = plan.Image is null ? null : plan.Strength,
            Inpainting = plan.Image is not null && plan.Mask is not null,
            Count = finished.Images.Count,
            Options = new Dictionary<string, string>(plan.Values) { ["seed"] = finished.Seed.ToString(CultureInfo.InvariantCulture) },
        };
        var shown = new List<GeneratedImageViewModel>();
        var files = new Dictionary<string, byte[]>();
        for (int i = 0; i < finished.Images.Count; i++)
        {
            var image = finished.Images[i];
            byte[] png = PngMetadata.Embed(ImageFiles.ToPng(image), Metadata(file, finished.Seed + i));
            var item = new GeneratedImageViewModel(i + 1, png, image, finished.Seed + i);
            shown.Add(item);
            files[item.FileName] = png;
        }

        files[ParametersName] = JsonSerializer.SerializeToUtf8Bytes(file, StudioJsonContext.Default.ImageParametersFile);
        var metadata = new Dictionary<string, string>
        {
            ["size"] = $"{first.Width} × {first.Height}",
            ["steps"] = file.Steps.ToString(CultureInfo.InvariantCulture),
            ["seed"] = file.Seed.ToString(CultureInfo.InvariantCulture),
            ["time"] = $"{seconds:0.0} s",
        };
        if (file.Sampler.Length > 0)
        {
            metadata["sampler"] = file.Sampler;
        }

        if (file.Inpainting)
        {
            metadata["mode"] = "inpainting";
        }
        else if (plan.Image is not null)
        {
            metadata["mode"] = "image to image";
        }

        if (shown.Count > 1)
        {
            metadata["images"] = shown.Count.ToString(CultureInfo.InvariantCulture);
        }

        string summary = string.Join(" · ", metadata.Select(pair => $"{pair.Key}: {pair.Value}"));
        Services.Dispatcher.Post(() =>
        {
            _last = file;
            Show(shown, summary);
        });
        Archived(NewRecord(title, model, plan, outcome, seconds, null, file, [.. shown.Select(item => item.FileName), ParametersName], metadata), files);
    }

    private RunRecord NewRecord(string title, ILoadedModel model, GenerationPlan plan, RunOutcome outcome, double seconds, string? error, ImageParametersFile? file,
        List<string> fileNames, Dictionary<string, string> metadata)
    {
        var input = new Dictionary<string, string> { ["prompt"] = plan.Prompt };
        if (plan.NegativePrompt.Length > 0)
        {
            input["negative prompt"] = plan.NegativePrompt;
        }

        if (plan.Image is not null)
        {
            input["input image"] = $"{plan.Image.Width} × {plan.Image.Height}";
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

    private static IEnumerable<KeyValuePair<string, string>> Metadata(ImageParametersFile file, long seed)
    {
        var text = new StringBuilder(file.Prompt);
        if (file.NegativePrompt.Length > 0)
        {
            text.Append("\nNegative prompt: ").Append(file.NegativePrompt);
        }

        text.Append("\nSteps: ").Append(file.Steps.ToString(CultureInfo.InvariantCulture))
            .Append(", Sampler: ").Append(file.Sampler.Length > 0 ? file.Sampler : "default")
            .Append(", CFG scale: ").Append(file.Guidance.ToString("0.0#", CultureInfo.InvariantCulture))
            .Append(", Seed: ").Append(seed.ToString(CultureInfo.InvariantCulture))
            .Append(", Size: ").Append(file.Width.ToString(CultureInfo.InvariantCulture)).Append('x').Append(file.Height.ToString(CultureInfo.InvariantCulture))
            .Append(", Model: ").Append(file.Model);
        if (file.Strength is { } strength)
        {
            text.Append(", Denoising strength: ").Append(strength.ToString("0.0#", CultureInfo.InvariantCulture));
        }

        yield return new("parameters", text.ToString());
        yield return new("Prompt", file.Prompt);
        if (file.NegativePrompt.Length > 0)
        {
            yield return new("Negative prompt", file.NegativePrompt);
        }

        yield return new("Seed", seed.ToString(CultureInfo.InvariantCulture));
        yield return new("Steps", file.Steps.ToString(CultureInfo.InvariantCulture));
        yield return new("Guidance", file.Guidance.ToString("0.0#", CultureInfo.InvariantCulture));
        yield return new("Sampler", file.Sampler);
        yield return new("Size", $"{file.Width}x{file.Height}");
        yield return new("Model", file.Model);
        yield return new("Software", "Sharplight Studio");
    }

    private void Show(List<GeneratedImageViewModel> images, string summary)
    {
        Images.Clear();
        foreach (var image in images)
        {
            Images.Add(image);
        }

        PreviewPng = null;
        StepText = "";
        ResultSummary = summary;
        SelectedImage = images.Count > 0 ? images[0] : null;
        OnPropertyChanged(nameof(HasSeveral));
    }

    /// <summary>Saves the shown image (with its metadata) where the user chooses.</summary>
    [RelayCommand]
    private async Task SaveImageAsync()
    {
        if (SelectedImage is { } image)
        {
            await Exporter.SaveAsync(Services.Platform, "Save image", $"sharplight-{image.Seed}.png", image.Png);
        }
    }

    /// <summary>Puts the shown image's seed into the parameters, so the next run starts from the same noise.</summary>
    [RelayCommand]
    private void ReuseSeed()
    {
        if (SelectedImage is { } image)
        {
            Panel.Parameters.OfType<TextParameterViewModel>().First(parameter => parameter.Key == "seed").Value = image.Seed.ToString(CultureInfo.InvariantCulture);
        }
    }

    /// <summary>Takes the shown image as the input image of the next run.</summary>
    [RelayCommand]
    private void UseAsInput()
    {
        if (SelectedImage is { } image)
        {
            Source.Set(image.Image, $"image-{image.Seed}.png");
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

    private static ParameterPanelViewModel<ImageGenerationOptions> CreateParameters(StudioServices services)
    {
        ParameterViewModel[] parameters =
        [
            new ChoiceParameterViewModel("size", "Size", "Pixels per side; the model's own size gives the best pictures, other sizes rebuild the model once.", s_sizes),
            new NumberParameterViewModel("steps", "Steps", "Denoising steps: more are slower and finer.", 1, 100, 1, GenericSteps),
            new NumberParameterViewModel("guidance", "Guidance", "How closely the picture follows the prompt (classifier-free guidance); distilled models ignore it.", 0, 20, 0.5, GenericGuidance),
            new ChoiceParameterViewModel("sampler", "Sampler", "The scheduler that steps the noise away.", s_samplers),
            new NumberParameterViewModel("strength", "Strength", "Image to image: how much of the input image is replaced (1 ignores it).", 0.05, 1, 0.05, 0.75),
            new TextParameterViewModel("seed", "Seed", "Empty picks a random seed; a number repeats the picture.", "Random"),
            new NumberParameterViewModel("count", "Images", "How many pictures to make (seeds follow each other).", 1, 8, 1, 1),
            new BoolParameterViewModel("preview", "Live preview", "Show the picture taking shape while it is generated.", defaultValue: true),
        ];
        ParameterPreset[] presets =
        [
            new("Fast", new() { ["steps"] = "12", ["guidance"] = "6", ["sampler"] = Samplers.DpmPlusPlus2MKarras }, BuiltIn: true),
            new("Quality", new() { ["steps"] = "35", ["guidance"] = "7.5", ["sampler"] = Samplers.DpmPlusPlus2MKarras }, BuiltIn: true),
        ];
        return new ParameterPanelViewModel<ImageGenerationOptions>(services.Pages, "image", parameters, panel =>
        {
            string size = panel.GetChoice("size");
            int? width = null;
            int? height = null;
            if (size != "native" && size.Split('x') is [var w, var h] && int.TryParse(w, out int parsedWidth) && int.TryParse(h, out int parsedHeight))
            {
                width = parsedWidth;
                height = parsedHeight;
            }

            return new ImageGenerationOptions
            {
                Width = width,
                Height = height,
                Steps = panel.GetInt("steps"),
                GuidanceScale = (float)panel.GetNumber("guidance"),
                Scheduler = panel.GetChoice("sampler") is { Length: > 0 } sampler ? sampler : null,
                Count = panel.GetInt("count"),
            };
        }, presets);
    }

    /// <summary>Everything one run needs, read from the page when it starts.</summary>
    private sealed record GenerationPlan(string Prompt, string NegativePrompt, ImageGenerationOptions Options, long? Seed, ImageInput? Image, ImageInput? Mask, float Strength,
        Dictionary<string, string> Values, bool Preview);
}
