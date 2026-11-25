using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// The built-in <see cref="IImageUpscaler"/> over the upscale contract: <see cref="WellKnown.UpscaleGraph"/> takes <c>pixels</c> [1, 3, H, W] (RGB, 0..1) of a fixed
/// size and returns the enlarged image. The graph for an image or tile size is rebuilt through the model's dimensions (<see cref="WellKnown.UpscaleWidthDimension"/>,
/// <see cref="WellKnown.UpscaleHeightDimension"/>) and kept. An image that fits one tile runs whole (no padding, exactly the network's output); a larger one runs
/// as windows of the tile size plus its padding that cover the image, shifted back inside it at the edges, whose outputs are blended with weights that fall to
/// zero towards the borders they share with a neighbour.
/// </summary>
internal sealed class ImageUpscalerClient : IImageUpscaler
{
    private const int CachedGraphs = 2;

    private readonly TaskModel _model;
    private readonly int _multiple;
    private readonly Dictionary<(int Width, int Height), CompiledModel> _compiled = [];
    private readonly Queue<(int Width, int Height)> _order = new();
    private readonly object _gate = new();

    public ImageUpscalerClient(TaskModel model)
    {
        _model = model;
        var config = model.Description.Component(ModelComponentRoles.VisionEncoder).Config;
        Scale = config["scale"].GetInt32();
        _multiple = config["multiple"].GetInt32();
        Metadata = TaskClientMetadata.For(model.Definition);
    }

    public TaskClientMetadata Metadata { get; }

    public int Scale { get; }

    public object? GetService(Type serviceType, object? serviceKey = null)
        => serviceKey is null && serviceType.IsInstanceOfType(_model) ? _model : null;

    public Task<ImageInput> UpscaleAsync(ImageInput image, ImageUpscalingOptions? options = null, IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        image.Validate();
        options ??= new ImageUpscalingOptions();
        ArgumentOutOfRangeException.ThrowIfNegative(options.TileSize);
        ArgumentOutOfRangeException.ThrowIfNegative(options.TilePadding);
        cancellationToken.ThrowIfCancellationRequested();
        return Task.Run(() =>
        {
            lock (_gate)
            {
                return Upscale(image, options, progress, cancellationToken);
            }
        }, cancellationToken);
    }

    public void Dispose()
    {
        lock (_gate)
        {
            foreach (var compiled in _compiled.Values)
            {
                compiled.Dispose();
            }

            _compiled.Clear();
        }
    }

    private ImageInput Upscale(ImageInput image, ImageUpscalingOptions options, IProgress<double>? progress, CancellationToken cancellationToken)
    {
        // The network wants multiples of its unshuffle factor: the missing edge pixels repeat the last ones and are cut off again.
        int sourceWidth = image.Width;
        int sourceHeight = image.Height;
        int width = RoundUp(sourceWidth, _multiple);
        int height = RoundUp(sourceHeight, _multiple);
        var planes = ToPlanes(image);
        if (width != sourceWidth || height != sourceHeight)
        {
            planes = Crop(planes, sourceWidth, sourceHeight, 0, 0, width, height);
        }

        int pad = options.TilePadding;
        int outWidth = width * Scale;
        int outHeight = height * Scale;
        float[] result;
        if (options.TileSize == 0 || (width <= options.TileSize && height <= options.TileSize))
        {
            result = Run(planes, width, height, cancellationToken);
            progress?.Report(1);
        }
        else
        {
            int canvasWidth = Math.Min(width, RoundUp(options.TileSize + (2 * pad), _multiple));
            int canvasHeight = Math.Min(height, RoundUp(options.TileSize + (2 * pad), _multiple));
            var xs = Starts(width, canvasWidth, pad);
            var ys = Starts(height, canvasHeight, pad);
            result = new float[3 * outWidth * outHeight];
            var weight = new float[outWidth * outHeight];
            int done = 0;
            foreach (int y0 in ys)
            {
                foreach (int x0 in xs)
                {
                    var tile = Run(Crop(planes, width, height, x0, y0, canvasWidth, canvasHeight), canvasWidth, canvasHeight, cancellationToken);
                    Blend(tile, canvasWidth * Scale, canvasHeight * Scale, x0 * Scale, y0 * Scale, outWidth, outHeight, result, weight,
                        x0 > 0, y0 > 0, x0 + canvasWidth < width, y0 + canvasHeight < height, pad * Scale);
                    progress?.Report((double)++done / (xs.Count * ys.Count));
                }
            }

            for (int c = 0; c < 3; c++)
            {
                for (int i = 0; i < weight.Length; i++)
                {
                    result[(c * weight.Length) + i] /= weight[i];
                }
            }
        }

        return ToImage(Crop(result, outWidth, outHeight, 0, 0, sourceWidth * Scale, sourceHeight * Scale), sourceWidth * Scale, sourceHeight * Scale);
    }

    // The window origins along one axis: windows of the canvas size that overlap by twice the padding, the last one shifted back to end at the edge.
    private static List<int> Starts(int length, int canvas, int pad)
    {
        var starts = new List<int>();
        int stride = Math.Max(1, canvas - 2 * pad);
        for (int start = 0; ; start += stride)
        {
            if (start + canvas >= length)
            {
                starts.Add(length - canvas);
                break;
            }

            starts.Add(start);
        }

        return starts;
    }

    // Adds a tile's output to the sums; the weight of a pixel rises linearly with its distance from every border that a neighbouring tile shares.
    private static void Blend(float[] tile, int tileWidth, int tileHeight, int x0, int y0, int width, int height, float[] sum, float[] weight,
        bool left, bool top, bool right, bool bottom, int pad)
    {
        int plane = width * height;
        int tilePlane = tileWidth * tileHeight;
        for (int y = 0; y < tileHeight; y++)
        {
            float wy = Ramp(y, tileHeight, top, bottom, pad);
            for (int x = 0; x < tileWidth; x++)
            {
                float w = wy * Ramp(x, tileWidth, left, right, pad);
                int target = ((y0 + y) * width) + x0 + x;
                weight[target] += w;
                for (int c = 0; c < 3; c++)
                {
                    sum[(c * plane) + target] += w * tile[(c * tilePlane) + (y * tileWidth) + x];
                }
            }
        }
    }

    private static float Ramp(int position, int length, bool before, bool after, int pad)
    {
        float w = 1;
        if (before)
        {
            w = Math.Min(w, (position + 0.5f) / Math.Max(1, pad));
        }

        if (after)
        {
            w = Math.Min(w, (length - position - 0.5f) / Math.Max(1, pad));
        }

        return Math.Max(w, 1e-4f);
    }

    private float[] Run(float[] input, int width, int height, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var compiled = Compiled(width, height);
        var pixels = compiled.Graph.Inputs.First(v => v.Input!.Name == "pixels");
        compiled.SetInput<float>(pixels, input);
        compiled.Run(new RunParameters());
        var output = new float[3 * width * Scale * height * Scale];
        compiled.GetOutput<float>(WellKnown.UpscaledOutput, output);
        return output;
    }

    private CompiledModel Compiled(int width, int height)
    {
        if (_compiled.TryGetValue((width, height), out var existing))
        {
            return existing;
        }

        while (_order.Count >= CachedGraphs)
        {
            var oldest = _order.Dequeue();
            _compiled.Remove(oldest, out var dropped);
            dropped?.Dispose();
        }

        var dimensions = new Dictionary<string, long>
        {
            [WellKnown.UpscaleWidthDimension] = width,
            [WellKnown.UpscaleHeightDimension] = height,
        };
        var compiled = _model.CompileComponent(ModelComponentRoles.VisionEncoder, WellKnown.UpscaleGraph, null, dimensions);
        _compiled[(width, height)] = compiled;
        _order.Enqueue((width, height));
        return compiled;
    }

    private static int RoundUp(int value, int multiple) => (value + multiple - 1) / multiple * multiple;

    private static float[] ToPlanes(ImageInput image)
    {
        int count = image.Width * image.Height;
        var planes = new float[3 * count];
        for (int i = 0; i < count; i++)
        {
            planes[i] = image.Rgb[3 * i] / 255f;
            planes[count + i] = image.Rgb[(3 * i) + 1] / 255f;
            planes[(2 * count) + i] = image.Rgb[(3 * i) + 2] / 255f;
        }

        return planes;
    }

    // The window [x0, x0 + width) × [y0, y0 + height) of three planes; pixels past the source edge repeat its last row or column.
    private static float[] Crop(float[] planes, int sourceWidth, int sourceHeight, int x0, int y0, int width, int height)
    {
        var result = new float[3 * width * height];
        for (int c = 0; c < 3; c++)
        {
            for (int y = 0; y < height; y++)
            {
                int sy = Math.Min(y0 + y, sourceHeight - 1);
                for (int x = 0; x < width; x++)
                {
                    int sx = Math.Min(x0 + x, sourceWidth - 1);
                    result[(c * width * height) + (y * width) + x] = planes[(c * sourceWidth * sourceHeight) + (sy * sourceWidth) + sx];
                }
            }
        }

        return result;
    }

    private static ImageInput ToImage(float[] planes, int width, int height)
    {
        int count = width * height;
        var rgb = new byte[3 * count];
        for (int i = 0; i < count; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                rgb[(3 * i) + c] = (byte)Math.Clamp((planes[(c * count) + i] * 255f) + 0.5f, 0f, 255f);
            }
        }

        return new ImageInput(width, height, rgb);
    }
}
