using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tokenizers;

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
    private readonly CompiledModel? _textCompiled;
    private readonly ITokenizer? _tokenizer;
    private readonly Graph.Value? _textTokens;
    private readonly Graph.Value? _poolIndex;
    private readonly int _contextLength;
    private readonly bool _poolEos;
    private readonly int _padId;
    private readonly int _textDimensions;
    private readonly SemaphoreSlim _gate = new(1, 1);

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
        if (model.Description.TryGetComponent(ModelComponentRoles.TextEncoder, out var text) && text.Graphs.Contains(WellKnown.TextEncodeGraph))
        {
            _tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
            _textCompiled = model.CompileComponent(ModelComponentRoles.TextEncoder, WellKnown.TextEncodeGraph);
            _textTokens = _textCompiled.Graph.Inputs.First(v => v.Input!.Name == "tokens");
            _poolIndex = _textCompiled.Graph.Inputs.First(v => v.Input!.Name == "pool_index");
            _contextLength = text.Config["contextLength"].GetInt32();
            _poolEos = text.Config["pool"].GetString() == "eos";
            _padId = text.Config.TryGetValue("padTokenId", out var pad) ? pad.GetInt32() : _tokenizer.PadId ?? 0;
            var shape = _textCompiled.Graph.Outputs[WellKnown.PooledOutput].Shape;
            _textDimensions = (int)shape[shape.Rank - 1].Fixed;
        }
    }

    public TaskClientMetadata Metadata { get; }

    public IReadOnlyList<string> Labels { get; }

    public int Dimensions { get; }

    public bool SupportsText => _textCompiled is not null;

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

        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var (_, logits) = Run(image);
            return Rank(logits, Labels, _sigmoid, options?.TopK ?? 5);
        }, cancellationToken);
    }

    public Task<float[]> EmbedImageAsync(ImageInput image, ImageEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        cancellationToken.ThrowIfCancellationRequested();
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var (pooled, _) = Run(image);
            if (options?.Normalize ?? true)
            {
                Normalize(pooled);
            }

            return pooled;
        }, cancellationToken);
    }

    public Task<float[]> EmbedTextAsync(string text, ImageEmbeddingOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        cancellationToken.ThrowIfCancellationRequested();
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var embedding = RunText(text);
            if (options?.Normalize ?? true)
            {
                Normalize(embedding);
            }

            return embedding;
        }, cancellationToken);
    }

    public Task<ImageClassification> ClassifyZeroShotAsync(ImageInput image, IReadOnlyList<string> candidates, ImageClassificationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(candidates);
        cancellationToken.ThrowIfCancellationRequested();
        RequireText();
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var (pooled, _) = Run(image);
            Normalize(pooled);
            var logits = new float[candidates.Count];
            for (int i = 0; i < candidates.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var text = RunText(candidates[i]);
                Normalize(text);
                double dot = 0;
                for (int d = 0; d < text.Length; d++)
                {
                    dot += (double)pooled[d] * text[d];
                }

                var (scale, bias) = ZeroShotScale();
                logits[i] = (float)(dot * scale + bias);
            }

            return Rank(logits, candidates, _sigmoid, options?.TopK ?? 5);
        }, cancellationToken);
    }

    public void Dispose()
    {
        _textCompiled?.Dispose();
        _compiled.Dispose();
        _gate.Dispose();
    }

    private void RequireText()
    {
        if (_textCompiled is null)
        {
            throw new NotSupportedException($"{Metadata.ModelName} has no text tower.");
        }
    }

    private (float Scale, float Bias) ZeroShotScale()
    {
        var graph = _textCompiled!.Graph;
        var scale = new float[1] { 1f };
        var bias = new float[1];
        if (graph.Outputs.ContainsKey("logit_scale"))
        {
            _textCompiled.GetOutput<float>("logit_scale", scale);
        }

        if (graph.Outputs.ContainsKey("logit_bias"))
        {
            _textCompiled.GetOutput<float>("logit_bias", bias);
        }

        return (scale[0], bias[0]);
    }

    private float[] RunText(string text)
    {
        RequireText();
        var ids = _tokenizer!.Encode(text);
        int eos = _tokenizer.EosId ?? -1;
        if (ids.Length > _contextLength)
        {
            ids = ids[.._contextLength];
            if (eos >= 0 && _tokenizer.AddEos)
            {
                ids[^1] = eos;
            }
        }

        var padded = new int[_contextLength];
        Array.Fill(padded, _padId);
        ids.CopyTo(padded, 0);
        int pool = _contextLength - 1;
        if (_poolEos)
        {
            pool = Array.IndexOf(ids, ids.Max());
        }

        _textCompiled!.SetInput<int>(_textTokens!, padded);
        _textCompiled.SetInput<int>(_poolIndex!, [pool]);
        _textCompiled.Run(new RunParameters());
        var pooled = new float[_textDimensions];
        _textCompiled.GetOutput<float>(WellKnown.PooledOutput, pooled);
        return pooled;
    }

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
