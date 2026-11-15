using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// The built-in implementation of <see cref="IImageClassifier"/> and <see cref="IImageEmbedder"/> over the image-encoder contract: a
/// <see cref="ModelComponentRoles.VisionEncoder"/> component whose <see cref="WellKnown.ImageEncodeGraph"/> takes the patchified image and
/// returns the pooled embedding (and class logits), a <see cref="ModelComponentRoles.FeatureExtractor"/> component with the
/// <see cref="ImageProcessorConfig"/>, and model-level <c>labels</c> / <c>scoring</c> configuration. Which model produced the graph is
/// irrelevant here.
/// </summary>
internal sealed class ImageEncoderClient : IImageClassifier, IImageEmbedder
{
    /// <summary>Component configuration key of the patch side the graph expects.</summary>
    public const string PatchSizeKey = "patchSize";

    /// <summary>Model configuration key of the class names (JSON array of strings).</summary>
    public const string LabelsKey = "labels";

    /// <summary>Model configuration key of the class scoring: <c>softmax</c> (default) or <c>sigmoid</c>.</summary>
    public const string ScoringKey = "scoring";

    private readonly TaskModel _model;
    private readonly CompiledModel _compiled;
    private readonly Graph.Value _patches;
    private readonly ImageProcessor _processor;
    private readonly int _patchSize;
    private readonly bool _sigmoid;
    private readonly bool _hasLogits;

    public ImageEncoderClient(TaskModel model)
    {
        _model = model;
        var encoder = model.Description.Component(ModelComponentRoles.VisionEncoder);
        _patchSize = encoder.Config.TryGetValue(PatchSizeKey, out var patch) ? patch.GetInt32() : throw new InvalidDataException($"The {ModelComponentRoles.VisionEncoder} component has no '{PatchSizeKey}'.");
        var extractor = model.Description.Component(ModelComponentRoles.FeatureExtractor);
        _processor = new ImageProcessor(ImageProcessorConfig.FromJson(extractor.Config[ImageProcessor.ComponentKey]));
        Labels = model.Description.Config.TryGetValue(LabelsKey, out var labels) ? [.. labels.EnumerateArray().Select(l => l.GetString() ?? "")] : [];
        _sigmoid = model.Description.ConfigValue(ScoringKey, "softmax") == "sigmoid";
        _compiled = model.CompileComponent(ModelComponentRoles.VisionEncoder, WellKnown.ImageEncodeGraph);
        _patches = _compiled.Graph.Inputs.First(v => v.Input!.Name == "patches");
        _hasLogits = _compiled.Graph.Outputs.ContainsKey(WellKnown.LogitsOutput);
        Dimensions = LastDim(WellKnown.PooledOutput);
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public IReadOnlyList<string> Labels { get; }

    public int Dimensions { get; }

    public bool SupportsText => false;

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<ImageClassification> ClassifyAsync(ImageInput image, ImageClassificationOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        cancellationToken.ThrowIfCancellationRequested();
        if (!_hasLogits)
        {
            throw new NotSupportedException($"{Metadata.ModelName} has no classifier head.");
        }

        var (_, logits) = Run(image);
        return Task.FromResult(Rank(logits, Labels, _sigmoid, options?.TopK ?? 5));
    }

    public Task<float[]> EmbedImageAsync(ImageInput image, ImageEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        cancellationToken.ThrowIfCancellationRequested();
        var (pooled, _) = Run(image);
        if (options?.Normalize ?? true)
        {
            Normalize(pooled);
        }

        return Task.FromResult(pooled);
    }

    public Task<float[]> EmbedTextAsync(string text, ImageEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
        => throw new NotSupportedException($"{Metadata.ModelName} has no text tower.");

    public Task<ImageClassification> ClassifyZeroShotAsync(ImageInput image, IReadOnlyList<string> candidates, ImageClassificationOptions? options = null,
        CancellationToken cancellationToken = default)
        => throw new NotSupportedException($"{Metadata.ModelName} has no text tower.");

    public void Dispose() => _compiled.Dispose();

    private int LastDim(string output)
    {
        var shape = _compiled.Graph.Outputs[output].Shape;
        return (int)shape[shape.Rank - 1].Fixed;
    }

    private (float[] Pooled, float[] Logits) Run(ImageInput image)
    {
        var processed = _processor.Process(image);
        _compiled.SetInput<float>(_patches, processed.Patchify(_patchSize));
        _compiled.Run(new RunParameters());
        var pooled = new float[Dimensions];
        _compiled.GetOutput<float>(WellKnown.PooledOutput, pooled);
        float[] logits = [];
        if (_hasLogits)
        {
            logits = new float[LastDim(WellKnown.LogitsOutput)];
            _compiled.GetOutput<float>(WellKnown.LogitsOutput, logits);
        }

        return (pooled, logits);
    }

    /// <summary>Scales <paramref name="vector"/> to unit length.</summary>
    internal static void Normalize(Span<float> vector)
    {
        double sum = 0;
        foreach (float v in vector)
        {
            sum += (double)v * v;
        }

        float scale = sum > 0 ? (float)(1.0 / Math.Sqrt(sum)) : 0f;
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] *= scale;
        }
    }

    /// <summary>Scores logits (softmax or sigmoid) and returns the best <paramref name="topK"/> classes.</summary>
    internal static ImageClassification Rank(float[] logits, IReadOnlyList<string> labels, bool sigmoid, int topK)
    {
        var scores = new float[logits.Length];
        if (sigmoid)
        {
            for (int i = 0; i < logits.Length; i++)
            {
                scores[i] = 1f / (1f + MathF.Exp(-logits[i]));
            }
        }
        else
        {
            float max = logits.Length > 0 ? logits.Max() : 0f;
            double sum = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                scores[i] = MathF.Exp(logits[i] - max);
                sum += scores[i];
            }

            for (int i = 0; i < logits.Length; i++)
            {
                scores[i] = (float)(scores[i] / sum);
            }
        }

        int count = topK <= 0 ? logits.Length : Math.Min(topK, logits.Length);
        var best = Enumerable.Range(0, logits.Length).OrderByDescending(i => scores[i]).ThenBy(i => i).Take(count)
            .Select(i => new ImageLabelScore(i, i < labels.Count ? labels[i] : $"class {i}", scores[i], logits[i]))
            .ToArray();
        return new ImageClassification(best);
    }
}
