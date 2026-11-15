using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// The built-in <see cref="IObjectDetector"/> over the detection contract: a <see cref="ModelComponentRoles.VisionEncoder"/> component whose
/// <see cref="WellKnown.DetectGraph"/> takes the normalized image <c>pixels</c> [1, 3, H, W] and returns per-query class <c>logits</c> [Q, C] and
/// <see cref="WellKnown.BoxesOutput"/> [Q, 4] (center x, center y, width, height as fractions of the model input), a
/// <see cref="ModelComponentRoles.FeatureExtractor"/> with the <see cref="ImageProcessorConfig"/>, and model-level <c>labels</c> and <c>scoring</c>
/// (<c>softmax-background</c>: softmax over the classes plus a last no-object class, DETR; <c>sigmoid</c>: independent scores, RT-DETR).
/// Which model produced the graph is irrelevant here.
/// </summary>
internal sealed class ObjectDetectorClient : IObjectDetector
{
    /// <summary>Model configuration key of the class scoring.</summary>
    public const string ScoringKey = "scoring";

    private readonly TaskModel _model;
    private readonly CompiledModel _compiled;
    private readonly Graph.Value _pixels;
    private readonly ImageProcessor _processor;
    private readonly bool _background;
    private readonly int _queries;
    private readonly int _classes;

    public ObjectDetectorClient(TaskModel model)
    {
        _model = model;
        var extractor = model.Description.Component(ModelComponentRoles.FeatureExtractor);
        _processor = new ImageProcessor(ImageProcessorConfig.FromJson(extractor.Config[ImageProcessor.ComponentKey]));
        Labels = model.Description.Config.TryGetValue(ImageEncoderClient.LabelsKey, out var labels) ? [.. labels.EnumerateArray().Select(l => l.GetString() ?? "")] : [];
        _background = model.Description.ConfigValue(ScoringKey, "sigmoid") == "softmax-background";
        _compiled = model.CompileComponent(ModelComponentRoles.VisionEncoder, WellKnown.DetectGraph);
        _pixels = _compiled.Graph.Inputs.First(v => v.Input!.Name == "pixels");
        var logits = _compiled.Graph.Outputs[WellKnown.LogitsOutput].Shape;
        _queries = (int)logits[0].Fixed;
        _classes = (int)logits[1].Fixed;
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public IReadOnlyList<string> Labels { get; }

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<ObjectDetection> DetectAsync(ImageInput image, ObjectDetectionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        cancellationToken.ThrowIfCancellationRequested();
        options ??= new ObjectDetectionOptions();
        var processed = _processor.Process(image);
        _compiled.SetInput<float>(_pixels, processed.Planes);
        _compiled.Run(new RunParameters());
        var logits = new float[_queries * _classes];
        var boxes = new float[_queries * 4];
        _compiled.GetOutput<float>(WellKnown.LogitsOutput, logits);
        _compiled.GetOutput<float>(WellKnown.BoxesOutput, boxes);
        return Task.FromResult(Decode(logits, boxes, image.Width, image.Height, options));
    }

    public void Dispose() => _compiled.Dispose();

    private ObjectDetection Decode(float[] logits, float[] boxes, int width, int height, ObjectDetectionOptions options)
    {
        var candidates = new List<DetectedObject>();
        int real = _background ? _classes - 1 : _classes;
        for (int q = 0; q < _queries; q++)
        {
            var row = logits.AsSpan(q * _classes, _classes);
            if (_background)
            {
                var (best, score) = BestSoftmax(row);
                if (best < real && score >= options.ScoreThreshold)
                {
                    candidates.Add(Make(best, score, boxes.AsSpan(q * 4, 4), width, height));
                }

                continue;
            }

            // Sigmoid heads score every class independently; a query may report more than one.
            for (int c = 0; c < real; c++)
            {
                float score = 1f / (1f + MathF.Exp(-row[c]));
                if (score >= options.ScoreThreshold)
                {
                    candidates.Add(Make(c, score, boxes.AsSpan(q * 4, 4), width, height));
                }
            }
        }

        var objects = candidates.OrderByDescending(o => o.Score).ToList();
        if (options.NmsIouThreshold is { } iou)
        {
            int[] keep = BoxUtilities.NonMaxSuppression([.. objects.Select(o => o.Box)], [.. objects.Select(o => o.Score)], iou,
                options.ClassAgnosticNms ? null : [.. objects.Select(o => o.ClassIndex)]);
            objects = [.. keep.Select(i => objects[i])];
        }

        if (options.MaxDetections > 0 && objects.Count > options.MaxDetections)
        {
            objects = objects.GetRange(0, options.MaxDetections);
        }

        return new ObjectDetection(objects);
    }

    private DetectedObject Make(int index, float score, ReadOnlySpan<float> box, int width, int height)
        => new(index, index < Labels.Count ? Labels[index] : $"class {index}",
            score, BoundingBox.FromCenter(box[0], box[1], box[2], box[3]).Scale(width, height).Clip(width, height));

    private static (int Index, float Score) BestSoftmax(ReadOnlySpan<float> row)
    {
        float max = float.NegativeInfinity;
        int best = 0;
        for (int i = 0; i < row.Length; i++)
        {
            if (row[i] > max)
            {
                max = row[i];
                best = i;
            }
        }

        double sum = 0;
        foreach (float v in row)
        {
            sum += Math.Exp(v - max);
        }

        return (best, (float)(1.0 / sum));
    }
}
