using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// The built-in <see cref="IObjectDetector"/> for dense detectors (the YOLO family): a <see cref="ModelComponentRoles.VisionEncoder"/> component whose
/// <see cref="WellKnown.DetectGraph"/> has one image input (normalized <c>[1, 3, H, W]</c>, letterboxed by the model's <see cref="ImageProcessorConfig"/>) and predicts for every grid cell, so
/// that the task layer applies the score threshold and the non-maximum suppression. The component's <c>denseHead</c> configuration names the output and its <c>layout</c>:
/// <c>channels-first</c> is <c>[4 + classes + masks, cells]</c> with center-size boxes in input pixels and class probabilities (instance segmentation models add the mask
/// coefficients and a second <c>protos</c> output <c>[masks, h, w]</c>), <c>end-to-end</c> is <c>[queries, 6]</c> with corner boxes in input pixels, the score and the class (NMS is part of the graph).
/// </summary>
internal sealed class DenseDetectorClient : IObjectDetector
{
    /// <summary>Component configuration key of the head description.</summary>
    public const string HeadKey = "denseHead";

    /// <summary>Layout of <c>[4 + classes + masks, cells]</c> outputs.</summary>
    public const string ChannelsFirst = "channels-first";

    /// <summary>Layout of <c>[queries, 6]</c> outputs (corners, score, class).</summary>
    public const string EndToEnd = "end-to-end";

    private const float DefaultIou = 0.7f;

    private readonly TaskModel _model;
    private readonly CompiledModel _compiled;
    private readonly Graph.Value _pixels;
    private readonly ImageProcessor _processor;
    private readonly string _layout;
    private readonly string _output;
    private readonly string? _protos;
    private readonly int _channels;
    private readonly int _cells;
    private readonly int _masks;
    private readonly int _protoHeight;
    private readonly int _protoWidth;
    private readonly SemaphoreSlim _gate = new(1, 1);

    public DenseDetectorClient(TaskModel model)
    {
        _model = model;
        var extractor = model.Description.Component(ModelComponentRoles.FeatureExtractor);
        _processor = new ImageProcessor(ImageProcessorConfig.FromJson(extractor.Config[ImageProcessor.ComponentKey]));
        Labels = model.Description.Config.TryGetValue(ImageEncoderClient.LabelsKey, out var labels) ? [.. labels.EnumerateArray().Select(l => l.GetString() ?? "")] : [];
        var component = model.Description.Component(ModelComponentRoles.VisionEncoder);
        var head = component.Config[HeadKey];
        _layout = head.GetProperty("layout").GetString() ?? ChannelsFirst;
        _output = head.GetProperty("output").GetString() ?? "output0";
        _protos = head.TryGetProperty("protos", out var protos) ? protos.GetString() : null;
        _compiled = model.CompileComponent(ModelComponentRoles.VisionEncoder, WellKnown.DetectGraph);
        _pixels = _compiled.Graph.Inputs[0];
        var shape = _compiled.Graph.Outputs[_output].Shape;
        if (_layout == EndToEnd)
        {
            _channels = (int)shape[-1].Fixed;
            _cells = (int)shape[-2].Fixed;
        }
        else
        {
            _channels = (int)shape[-2].Fixed;
            _cells = (int)shape[-1].Fixed;
        }

        if (_protos is not null)
        {
            var protoShape = _compiled.Graph.Outputs[_protos].Shape;
            _masks = (int)protoShape[-3].Fixed;
            _protoHeight = (int)protoShape[-2].Fixed;
            _protoWidth = (int)protoShape[-1].Fixed;
        }

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
        var settings = options ?? new ObjectDetectionOptions();
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var processed = _processor.Process(image);
            _compiled.SetInput<float>(_pixels, processed.Planes);
            var run = new RunParameters();
            if (_pixels.Shape.Symbol is { } batch)
            {
                // A dynamic batch axis (the ONNX batch symbol) runs with one image.
                run.Symbols[batch] = 1;
            }

            _compiled.Run(run);
            var raw = new float[_channels * _cells];
            _compiled.GetOutput<float>(_output, raw);
            float[]? protoValues = null;
            if (_protos is not null)
            {
                protoValues = new float[_masks * _protoHeight * _protoWidth];
                _compiled.GetOutput<float>(_protos, protoValues);
            }

            return _layout == EndToEnd ? DecodeEndToEnd(raw, processed, image, settings) : DecodeChannelsFirst(raw, protoValues, processed, image, settings);
        }, cancellationToken);
    }

    public void Dispose()
    {
        _compiled.Dispose();
        _gate.Dispose();
    }

    private ObjectDetection DecodeChannelsFirst(float[] raw, float[]? protos, ProcessedImage processed, ImageInput image, ObjectDetectionOptions options)
    {
        int classes = _channels - 4 - _masks;
        var boxes = new List<BoundingBox>();
        var scores = new List<float>();
        var classIndices = new List<int>();
        var cellIndices = new List<int>();
        for (int a = 0; a < _cells; a++)
        {
            int best = 0;
            float score = raw[4 * _cells + a];
            for (int c = 1; c < classes; c++)
            {
                float v = raw[(4 + c) * _cells + a];
                if (v > score)
                {
                    score = v;
                    best = c;
                }
            }

            if (score < options.ScoreThreshold)
            {
                continue;
            }

            boxes.Add(BoundingBox.FromCenter(raw[a], raw[_cells + a], raw[2 * _cells + a], raw[3 * _cells + a]));
            scores.Add(score);
            classIndices.Add(best);
            cellIndices.Add(a);
        }

        int[] keep = BoxUtilities.NonMaxSuppression(boxes, scores, options.NmsIouThreshold ?? DefaultIou, options.ClassAgnosticNms ? null : classIndices);
        if (options.MaxDetections > 0 && keep.Length > options.MaxDetections)
        {
            keep = keep[..options.MaxDetections];
        }

        var objects = new List<DetectedObject>(keep.Length);
        foreach (int k in keep)
        {
            SegmentMask? mask = null;
            if (protos is not null)
            {
                var coefficients = new float[_masks];
                for (int m = 0; m < _masks; m++)
                {
                    coefficients[m] = raw[(4 + classes + m) * _cells + cellIndices[k]];
                }

                mask = BuildMask(coefficients, protos, boxes[k], processed, image, scores[k]);
            }

            objects.Add(Make(classIndices[k], scores[k], boxes[k], processed, image, mask));
        }

        return new ObjectDetection(objects);
    }

    private ObjectDetection DecodeEndToEnd(float[] raw, ProcessedImage processed, ImageInput image, ObjectDetectionOptions options)
    {
        var objects = new List<DetectedObject>();
        for (int q = 0; q < _cells; q++)
        {
            var row = raw.AsSpan(q * _channels, _channels);
            if (row[4] >= options.ScoreThreshold)
            {
                objects.Add(Make((int)row[5], row[4], new BoundingBox(row[0], row[1], row[2], row[3]), processed, image, null));
            }
        }

        objects = [.. objects.OrderByDescending(o => o.Score)];
        if (options.MaxDetections > 0 && objects.Count > options.MaxDetections)
        {
            objects = objects.GetRange(0, options.MaxDetections);
        }

        return new ObjectDetection(objects);
    }

    private DetectedObject Make(int index, float score, BoundingBox inputBox, ProcessedImage processed, ImageInput image, SegmentMask? mask)
    {
        // Model input coordinates to source pixels: undo the crop / padding offset, then the scale.
        var box = new BoundingBox((inputBox.X1 + processed.OffsetX) / processed.ScaleX, (inputBox.Y1 + processed.OffsetY) / processed.ScaleY,
            (inputBox.X2 + processed.OffsetX) / processed.ScaleX, (inputBox.Y2 + processed.OffsetY) / processed.ScaleY);
        return new DetectedObject(index, index < Labels.Count ? Labels[index] : $"class {index}", score, box.Clip(image.Width, image.Height), mask);
    }

    /// <summary>The mask of one detection: the coefficients over the prototypes, sampled bilinearly (as the logit map upsampled to the input) at every source pixel, cut at the box.</summary>
    private SegmentMask BuildMask(float[] coefficients, float[] protos, BoundingBox inputBox, ProcessedImage processed, ImageInput image, float score)
    {
        int plane = _protoHeight * _protoWidth;
        var logits = new float[plane];
        for (int m = 0; m < _masks; m++)
        {
            float c = coefficients[m];
            var proto = protos.AsSpan(m * plane, plane);
            for (int i = 0; i < plane; i++)
            {
                logits[i] += c * proto[i];
            }
        }

        var pixels = new bool[image.Width * image.Height];
        float gx = (float)_protoWidth / processed.Width;
        float gy = (float)_protoHeight / processed.Height;
        for (int y = 0; y < image.Height; y++)
        {
            // Model input coordinate of the source pixel's center.
            float v = (y + 0.5f) * processed.ScaleY - processed.OffsetY;
            if (v < inputBox.Y1 || v >= inputBox.Y2)
            {
                continue;
            }

            float py = Math.Clamp(v * gy - 0.5f, 0, _protoHeight - 1);
            int y0 = (int)py;
            int y1 = Math.Min(y0 + 1, _protoHeight - 1);
            float fy = py - y0;
            for (int x = 0; x < image.Width; x++)
            {
                float u = (x + 0.5f) * processed.ScaleX - processed.OffsetX;
                if (u < inputBox.X1 || u >= inputBox.X2)
                {
                    continue;
                }

                float px = Math.Clamp(u * gx - 0.5f, 0, _protoWidth - 1);
                int x0 = (int)px;
                int x1 = Math.Min(x0 + 1, _protoWidth - 1);
                float fx = px - x0;
                float top = logits[y0 * _protoWidth + x0] * (1 - fx) + logits[y0 * _protoWidth + x1] * fx;
                float bottom = logits[y1 * _protoWidth + x0] * (1 - fx) + logits[y1 * _protoWidth + x1] * fx;
                pixels[y * image.Width + x] = top * (1 - fy) + bottom * fy > 0;
            }
        }

        return new SegmentMask(image.Width, image.Height, pixels, score);
    }
}
