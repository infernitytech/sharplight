using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// The built-in <see cref="ISemanticSegmenter"/> over the semantic segmentation contract: a <see cref="ModelComponentRoles.VisionEncoder"/> component whose
/// <see cref="WellKnown.SemanticSegmentGraph"/> takes the normalized image and returns class scores [1, K, h, w], a <see cref="ModelComponentRoles.FeatureExtractor"/>
/// with the <see cref="ImageProcessorConfig"/> and the model-level <c>labels</c>. A single class channel is a matte (sigmoid). Which model produced the graph is irrelevant here.
/// </summary>
internal sealed class SemanticSegmenterClient : ISemanticSegmenter
{
    private readonly TaskModel _model;
    private readonly CompiledModel _compiled;
    private readonly Graph.Value _pixels;
    private readonly ImageProcessor _processor;
    private readonly int _classes;
    private readonly int _width;
    private readonly int _height;
    private readonly SemaphoreSlim _gate = new(1, 1);

    public SemanticSegmenterClient(TaskModel model)
    {
        _model = model;
        var extractor = model.Description.Component(ModelComponentRoles.FeatureExtractor);
        _processor = new ImageProcessor(ImageProcessorConfig.FromJson(extractor.Config[ImageProcessor.ComponentKey]));
        _compiled = model.CompileComponent(ModelComponentRoles.VisionEncoder, WellKnown.SemanticSegmentGraph);
        _pixels = _compiled.Graph.Inputs.First(v => v.Input!.Name == "pixels");
        var shape = _compiled.Graph.Outputs[WellKnown.SemanticLogitsOutput].Shape;
        _classes = (int)shape[1].Fixed;
        _height = (int)shape[2].Fixed;
        _width = (int)shape[3].Fixed;
        ClassNames = model.Description.Config.TryGetValue(ImageEncoderClient.LabelsKey, out var labels) ? [.. labels.EnumerateArray().Select(l => l.GetString() ?? "")] : [];
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public IReadOnlyList<string> ClassNames { get; }

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<SemanticSegmentation> SegmentAsync(ImageInput image, SemanticSegmentationOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        cancellationToken.ThrowIfCancellationRequested();
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var processed = _processor.Process(image);
            _compiled.SetInput<float>(_pixels, processed.Planes);
            _compiled.Run(new RunParameters());
            var logits = new float[_classes * _width * _height];
            _compiled.GetOutput<float>(WellKnown.SemanticLogitsOutput, logits);
            bool resize = options?.ResizeToInput ?? true;
            int width = resize ? image.Width : _width;
            int height = resize ? image.Height : _height;
            var labels = new int[width * height];
            float[]? alpha = null;
            if (_classes == 1)
            {
                alpha = Sigmoid(resize ? PlaneResampling.Bilinear(logits, _width, _height, width, height) : logits);
                for (int i = 0; i < labels.Length; i++)
                {
                    labels[i] = alpha[i] > 0.5f ? 1 : 0;
                }
            }
            else
            {
                ArgMax(logits, width, height, labels);
            }

            return new SemanticSegmentation(width, height, labels, ClassNames)
            {
                Logits = options?.IncludeLogits == true ? logits : null,
                LogitsWidth = _width,
                LogitsHeight = _height,
                ClassCount = _classes,
                Alpha = alpha,
            };
        }, cancellationToken);
    }

    public void Dispose()
    {
        _compiled.Dispose();
        _gate.Dispose();
    }

    private static float[] Sigmoid(float[] values)
    {
        var result = new float[values.Length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = 1f / (1f + MathF.Exp(-values[i]));
        }

        return result;
    }

    // Bilinear resampling (half-pixel centers) of every class plane to the target size fused with the arg-max over classes, row-parallel.
    private void ArgMax(float[] logits, int width, int height, int[] labels)
    {
        int sw = _width;
        int sh = _height;
        int plane = sw * sh;
        var x0 = new int[width];
        var x1 = new int[width];
        var wx = new float[width];
        for (int x = 0; x < width; x++)
        {
            float fx = Math.Max(0, (x + 0.5f) * sw / width - 0.5f);
            x0[x] = Math.Min((int)fx, sw - 1);
            x1[x] = Math.Min(x0[x] + 1, sw - 1);
            wx[x] = fx - x0[x];
        }

        Parallel.For(0, height, y =>
        {
            float fy = Math.Max(0, (y + 0.5f) * sh / height - 0.5f);
            int y0 = Math.Min((int)fy, sh - 1);
            int y1 = Math.Min(y0 + 1, sh - 1);
            float wy = fy - y0;
            for (int x = 0; x < width; x++)
            {
                float best = float.NegativeInfinity;
                int label = 0;
                for (int c = 0; c < _classes; c++)
                {
                    int o = c * plane;
                    float top = logits[o + y0 * sw + x0[x]] * (1 - wx[x]) + logits[o + y0 * sw + x1[x]] * wx[x];
                    float bottom = logits[o + y1 * sw + x0[x]] * (1 - wx[x]) + logits[o + y1 * sw + x1[x]] * wx[x];
                    float v = top * (1 - wy) + bottom * wy;
                    if (v > best)
                    {
                        best = v;
                        label = c;
                    }
                }

                labels[y * width + x] = label;
            }
        });
    }
}
