using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// The built-in <see cref="IObjectDetector"/> for document layout detectors: the detection contract of <see cref="ObjectDetectorClient"/> (sigmoid scores) and, named by the
/// component's <c>layoutHead</c> configuration, three more outputs: <c>order</c> (pairwise reading-order logits [Q, Q]: entry [i, j] above the diagonal is the logit that query i is read
/// before query j), <c>maskEmbeddings</c> [Q, P] and <c>maskFeatures</c> [P, h·w] (a query's mask logits are its embedding times the features, positive inside). The objects come back
/// in reading order: every query's rank is its number of pairwise votes (as the reference post-processing computes it), and a detection's mask is the query's mask inside its box, at the size of the image.
/// </summary>
internal sealed class LayoutDetectorClient : IObjectDetector
{
    /// <summary>Component configuration key of the head description.</summary>
    public const string HeadKey = "layoutHead";

    private readonly TaskModel _model;
    private readonly CompiledModel _compiled;
    private readonly Graph.Value _pixels;
    private readonly ImageProcessor _processor;
    private readonly string _order;
    private readonly string _embeddings;
    private readonly string _features;
    private readonly int _queries;
    private readonly int _classes;
    private readonly int _prototypes;
    private readonly int _side;
    private readonly SemaphoreSlim _gate = new(1, 1);

    public LayoutDetectorClient(TaskModel model)
    {
        _model = model;
        var extractor = model.Description.Component(ModelComponentRoles.FeatureExtractor);
        _processor = new ImageProcessor(ImageProcessorConfig.FromJson(extractor.Config[ImageProcessor.ComponentKey]));
        Labels = model.Description.Config.TryGetValue(ImageEncoderClient.LabelsKey, out var labels) ? [.. labels.EnumerateArray().Select(l => l.GetString() ?? "")] : [];
        var head = model.Description.Component(ModelComponentRoles.VisionEncoder).Config[HeadKey];
        _order = head.GetProperty("order").GetString()!;
        _embeddings = head.GetProperty("maskEmbeddings").GetString()!;
        _features = head.GetProperty("maskFeatures").GetString()!;
        _compiled = model.CompileComponent(ModelComponentRoles.VisionEncoder, WellKnown.DetectGraph);
        _pixels = _compiled.Graph.Inputs.First(v => v.Input!.Name == "pixels");
        var logits = _compiled.Graph.Outputs[WellKnown.LogitsOutput].Shape;
        _queries = (int)logits[0].Fixed;
        _classes = (int)logits[1].Fixed;
        var features = _compiled.Graph.Outputs[_features].Shape;
        _prototypes = (int)features[0].Fixed;
        _side = (int)Math.Round(Math.Sqrt(features[1].Fixed));
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public IReadOnlyList<string> Labels { get; }

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<ObjectDetection> DetectAsync(ImageInput image, ObjectDetectionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        image.Validate();
        cancellationToken.ThrowIfCancellationRequested();
        var settings = options ?? new ObjectDetectionOptions();
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var processed = _processor.Process(image);
            _compiled.SetInput<float>(_pixels, processed.Planes);
            _compiled.Run(new RunParameters());
            var logits = new float[_queries * _classes];
            var boxes = new float[_queries * 4];
            var order = new float[_queries * _queries];
            _compiled.GetOutput<float>(WellKnown.LogitsOutput, logits);
            _compiled.GetOutput<float>(WellKnown.BoxesOutput, boxes);
            _compiled.GetOutput<float>(_order, order);
            var ranks = Ranks(order);
            var found = new List<(int Query, int Class, float Score)>();
            for (int q = 0; q < _queries; q++)
            {
                for (int c = 0; c < _classes; c++)
                {
                    float score = 1f / (1f + MathF.Exp(-logits[(q * _classes) + c]));
                    if (score >= settings.ScoreThreshold)
                    {
                        found.Add((q, c, score));
                    }
                }
            }

            var sorted = found.OrderBy(f => ranks[f.Query]).ThenByDescending(f => f.Score).ToList();
            if (settings.MaxDetections > 0 && sorted.Count > settings.MaxDetections)
            {
                sorted = sorted.GetRange(0, settings.MaxDetections);
            }

            float[]? embeddings = null;
            float[]? features = null;
            if (settings.IncludeMasks && sorted.Count > 0)
            {
                embeddings = new float[_queries * _prototypes];
                features = new float[_prototypes * _side * _side];
                _compiled.GetOutput<float>(_embeddings, embeddings);
                _compiled.GetOutput<float>(_features, features);
            }

            var objects = new List<DetectedObject>(sorted.Count);
            foreach (var (query, cls, score) in sorted)
            {
                var box = BoundingBox.FromCenter(boxes[query * 4], boxes[(query * 4) + 1], boxes[(query * 4) + 2], boxes[(query * 4) + 3]).Scale(image.Width, image.Height).Clip(image.Width, image.Height);
                var mask = embeddings is null ? null : BuildMask(embeddings.AsSpan(query * _prototypes, _prototypes), features!, box, image, score);
                objects.Add(new DetectedObject(cls, cls < Labels.Count ? Labels[cls] : $"class {cls}", score, box, mask, objects.Count));
            }

            return new ObjectDetection(objects);
        }, cancellationToken);
    }

    public void Dispose()
    {
        _compiled.Dispose();
        _gate.Dispose();
    }

    // The reading rank of every query: the votes of the pairwise logits (query j is after i when sigmoid(order[i, j]) is high), ascending; ties by index.
    private int[] Ranks(float[] order)
    {
        var votes = new double[_queries];
        for (int i = 0; i < _queries; i++)
        {
            for (int j = i + 1; j < _queries; j++)
            {
                double s = 1.0 / (1.0 + Math.Exp(-order[(i * _queries) + j]));
                votes[j] += s;
                votes[i] += 1.0 - s;
            }
        }

        var byVotes = Enumerable.Range(0, _queries).OrderBy(i => votes[i]).ThenBy(i => i).ToArray();
        var ranks = new int[_queries];
        for (int rank = 0; rank < byVotes.Length; rank++)
        {
            ranks[byVotes[rank]] = rank;
        }

        return ranks;
    }

    // The query's mask logits on the prototype map (positive inside), carried to the pixels of the image inside the box by nearest sampling.
    private SegmentMask BuildMask(ReadOnlySpan<float> embedding, float[] features, BoundingBox box, ImageInput image, float score)
    {
        int cells = _side * _side;
        var logits = new float[cells];
        for (int p = 0; p < _prototypes; p++)
        {
            float e = embedding[p];
            var row = features.AsSpan(p * cells, cells);
            for (int i = 0; i < cells; i++)
            {
                logits[i] += e * row[i];
            }
        }

        var pixels = new bool[image.Width * image.Height];
        int x0 = Math.Max(0, (int)MathF.Floor(box.X1));
        int x1 = Math.Min(image.Width, (int)MathF.Ceiling(box.X2));
        int y0 = Math.Max(0, (int)MathF.Floor(box.Y1));
        int y1 = Math.Min(image.Height, (int)MathF.Ceiling(box.Y2));
        for (int y = y0; y < y1; y++)
        {
            int row = Math.Min(_side - 1, (int)((long)y * _side / image.Height)) * _side;
            for (int x = x0; x < x1; x++)
            {
                pixels[(y * image.Width) + x] = logits[row + Math.Min(_side - 1, (int)((long)x * _side / image.Width))] > 0f;
            }
        }

        return new SegmentMask(image.Width, image.Height, pixels, score);
    }
}
