using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// The built-in <see cref="IImageSegmenter"/> over the promptable segmentation contract: <see cref="WellKnown.SegmentEncodeGraph"/> (image stage) and
/// <see cref="WellKnown.SegmentDecodeGraph"/> (prompt stage), a <see cref="ModelComponentRoles.FeatureExtractor"/> with the <see cref="ImageProcessorConfig"/>.
/// The candidate masks are the decoder's low-resolution logits, resampled bilinearly (through the model input size, as the reference does) to the
/// image and thresholded at 0. Which model produced the graphs is irrelevant here.
/// </summary>
internal sealed class ImageSegmenterClient : IImageSegmenter
{
    // The stability thresholds of the single-mask choice (SAM 2's dynamic multimask output).
    private const float StabilityDelta = 0.05f;
    private const float StabilityThreshold = 0.98f;

    private readonly TaskModel _model;
    private readonly CompiledModel _encoder;
    private readonly CompiledModel _decoder;
    private readonly ImageProcessor _processor;
    private readonly Graph.Value _pixels;
    private readonly Graph.Value _embedding;
    private readonly Graph.Value _features0;
    private readonly Graph.Value _features1;
    private readonly Graph.Value _coords;
    private readonly Graph.Value _kinds;
    private readonly int _inputSize;
    private readonly int _maskSize;
    private readonly int _maskCount;
    private readonly int _maxPromptTokens;
    private readonly object _gate = new();

    public ImageSegmenterClient(TaskModel model)
    {
        _model = model;
        var extractor = model.Description.Component(ModelComponentRoles.FeatureExtractor);
        _processor = new ImageProcessor(ImageProcessorConfig.FromJson(extractor.Config[ImageProcessor.ComponentKey]));
        var decoder = model.Description.Component(ModelComponentRoles.MaskDecoder);
        _inputSize = decoder.Config["imageSize"].GetInt32();
        _maskSize = decoder.Config["maskSize"].GetInt32();
        _maskCount = decoder.Config["maskCount"].GetInt32();
        _maxPromptTokens = decoder.Config["maxPromptTokens"].GetInt32();
        _encoder = model.CompileComponent(ModelComponentRoles.VisionEncoder, WellKnown.SegmentEncodeGraph);
        _decoder = model.CompileComponent(ModelComponentRoles.MaskDecoder, WellKnown.SegmentDecodeGraph);
        _pixels = Input(_encoder, "pixels");
        _embedding = Input(_decoder, WellKnown.SegmentEmbeddingOutput);
        _features0 = Input(_decoder, WellKnown.SegmentFeatures0Output);
        _features1 = Input(_decoder, WellKnown.SegmentFeatures1Output);
        _coords = Input(_decoder, "point_coords");
        _kinds = Input(_decoder, "point_kinds");
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<SegmentationImage> EncodeImageAsync(ImageInput image, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        image.Validate();
        cancellationToken.ThrowIfCancellationRequested();
        return Task.Run(() =>
        {
            lock (_gate)
            {
                var processed = _processor.Process(image);
                _encoder.SetInput<float>(_pixels, processed.Planes);
                _encoder.Run(new RunParameters());
                return new SegmentationImage(this, image.Width, image.Height, Read(_encoder, WellKnown.SegmentEmbeddingOutput), Read(_encoder, WellKnown.SegmentFeatures0Output),
                    Read(_encoder, WellKnown.SegmentFeatures1Output));
            }
        }, cancellationToken);
    }

    public Task<ImageSegmentation> SegmentAsync(SegmentationImage image, SegmentationPrompt prompt, SegmentationOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(prompt);
        if (!ReferenceEquals(image.Owner, this))
        {
            throw new ArgumentException("The image was encoded by another segmenter.", nameof(image));
        }

        if (prompt.Points.Count == 0 && prompt.Box is null)
        {
            throw new ArgumentException("The prompt has neither points nor a box.", nameof(prompt));
        }

        if (prompt.Points.Count + 3 > _maxPromptTokens)
        {
            throw new ArgumentException($"A prompt can have at most {_maxPromptTokens - 3} points.", nameof(prompt));
        }

        options ??= new SegmentationOptions();
        cancellationToken.ThrowIfCancellationRequested();
        return Task.Run(() =>
        {
            lock (_gate)
            {
                return Decode(image, prompt, options);
            }
        }, cancellationToken);
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _decoder.Dispose();
    }

    private ImageSegmentation Decode(SegmentationImage image, SegmentationPrompt prompt, SegmentationOptions options)
    {
        float sx = (float)_inputSize / image.Width;
        float sy = (float)_inputSize / image.Height;
        var coords = new List<float>();
        var kinds = new List<int>();
        foreach (var point in prompt.Points)
        {
            coords.Add(point.X * sx);
            coords.Add(point.Y * sy);
            kinds.Add(point.IsPositive ? SegmentPromptKinds.Positive : SegmentPromptKinds.Negative);
        }

        if (prompt.Box is { } box)
        {
            coords.AddRange([box.X1 * sx, box.Y1 * sy, box.X2 * sx, box.Y2 * sy, 0f, 0f]);
            kinds.AddRange([SegmentPromptKinds.BoxStart, SegmentPromptKinds.BoxEnd, SegmentPromptKinds.None]);
        }
        else
        {
            coords.AddRange([0f, 0f]);
            kinds.Add(SegmentPromptKinds.None);
        }

        _decoder.SetInput<float>(_embedding, image.Embedding);
        _decoder.SetInput<float>(_features0, image.Features0);
        _decoder.SetInput<float>(_features1, image.Features1);
        _decoder.SetInput<float>(_coords, coords.ToArray());
        _decoder.SetInput<int>(_kinds, kinds.ToArray());
        var run = new RunParameters();
        run.Symbols[WellKnown.SegmentPointSymbol] = kinds.Count;
        _decoder.Run(run);
        int plane = _maskSize * _maskSize;
        var logits = new float[_maskCount * plane];
        var scores = new float[_maskCount];
        var objectScore = new float[1];
        _decoder.GetOutput<float>(WellKnown.SegmentMasksOutput, logits);
        _decoder.GetOutput<float>(WellKnown.SegmentScoresOutput, scores);
        _decoder.GetOutput<float>(WellKnown.SegmentObjectOutput, objectScore);

        var chosen = options.MultipleMasks ? Enumerable.Range(1, _maskCount - 1).ToArray() : [SingleMask(logits, scores)];
        var masks = new List<SegmentMask>();
        foreach (int row in chosen)
        {
            masks.Add(ToMask(logits.AsSpan(row * plane, plane).ToArray(), scores[row], image.Width, image.Height));
        }

        return new ImageSegmentation(masks, objectScore[0]);
    }

    // The single-mask candidate (row 0) when its mask is stable, else the multi-mask candidate with the highest predicted IoU.
    private int SingleMask(float[] logits, float[] scores)
    {
        int plane = _maskSize * _maskSize;
        int inner = 0;
        int outer = 0;
        for (int i = 0; i < plane; i++)
        {
            float v = logits[i];
            inner += v > StabilityDelta ? 1 : 0;
            outer += v > -StabilityDelta ? 1 : 0;
        }

        float stability = outer > 0 ? (float)inner / outer : 1f;
        if (stability >= StabilityThreshold)
        {
            return 0;
        }

        int best = 1;
        for (int i = 2; i < _maskCount; i++)
        {
            best = scores[i] > scores[best] ? i : best;
        }

        return best;
    }

    private SegmentMask ToMask(float[] lowRes, float score, int width, int height)
    {
        var square = PlaneResampling.Bilinear(lowRes, _maskSize, _maskSize, _inputSize, _inputSize);
        var full = PlaneResampling.Bilinear(square, _inputSize, _inputSize, width, height);
        var pixels = new bool[full.Length];
        for (int i = 0; i < full.Length; i++)
        {
            pixels[i] = full[i] > 0;
        }

        return new SegmentMask(width, height, pixels, score);
    }

    private static Graph.Value Input(CompiledModel model, string name) => model.Graph.Inputs.First(v => v.Input!.Name == name);

    private static float[] Read(CompiledModel model, string output)
    {
        var shape = model.Graph.Outputs[output].Shape;
        long count = 1;
        for (int i = 0; i < shape.Rank; i++)
        {
            count *= shape[i].Fixed;
        }

        var data = new float[count];
        model.GetOutput<float>(output, data);
        return data;
    }
}
