using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// The built-in <see cref="IDepthEstimator"/> over the depth contract: a <see cref="ModelComponentRoles.VisionEncoder"/> component whose
/// <see cref="WellKnown.DepthGraph"/> takes the patchified image and returns the depth map [1, 1, H, W], and a
/// <see cref="ModelComponentRoles.FeatureExtractor"/> with the <see cref="ImageProcessorConfig"/>. Which model produced the graph is irrelevant here.
/// </summary>
internal sealed class DepthEstimatorClient : IDepthEstimator
{
    private readonly TaskModel _model;
    private readonly CompiledModel _compiled;
    private readonly Graph.Value _patches;
    private readonly ImageProcessor _processor;
    private readonly int _patchSize;
    private readonly int _width;
    private readonly int _height;
    private readonly SemaphoreSlim _gate = new(1, 1);

    public DepthEstimatorClient(TaskModel model)
    {
        _model = model;
        var encoder = model.Description.Component(ModelComponentRoles.VisionEncoder);
        _patchSize = encoder.Config[ImageEncoderClient.PatchSizeKey].GetInt32();
        var extractor = model.Description.Component(ModelComponentRoles.FeatureExtractor);
        _processor = new ImageProcessor(ImageProcessorConfig.FromJson(extractor.Config[ImageProcessor.ComponentKey]));
        _compiled = model.CompileComponent(ModelComponentRoles.VisionEncoder, WellKnown.DepthGraph);
        _patches = _compiled.Graph.Inputs.First(v => v.Input!.Name == "patches");
        var shape = _compiled.Graph.Outputs[WellKnown.DepthOutput].Shape;
        _height = (int)shape[2].Fixed;
        _width = (int)shape[3].Fixed;
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<DepthMap> EstimateAsync(ImageInput image, DepthEstimationOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        cancellationToken.ThrowIfCancellationRequested();
        return BackgroundCompute.RunAsync(_gate, () =>
        {
            var processed = _processor.Process(image);
            _compiled.SetInput<float>(_patches, processed.Patchify(_patchSize));
            _compiled.Run(new RunParameters());
            var depth = new float[_width * _height];
            _compiled.GetOutput<float>(WellKnown.DepthOutput, depth);
            if (options?.ResizeToInput ?? true)
            {
                return new DepthMap(image.Width, image.Height, PlaneResampling.Bilinear(depth, _width, _height, image.Width, image.Height));
            }

            return new DepthMap(_width, _height, depth);
        }, cancellationToken);
    }

    public void Dispose()
    {
        _compiled.Dispose();
        _gate.Dispose();
    }
}
