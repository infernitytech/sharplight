using Sharplight.Imaging;

namespace Sharplight.Vision.Tests;

/// <summary>The center crop's offsets: truncated like Hugging Face's processors, or rounded to the nearest integer (ties to even) like torchvision's <c>CenterCrop</c>.</summary>
public class CenterCropTests
{
    // Pixel (x, y) holds x in the red channel; the crop's first column therefore tells the offset.
    private static ImageInput Gradient(int width, int height)
    {
        var rgb = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                rgb[((y * width) + x) * 3] = (byte)x;
                rgb[(((y * width) + x) * 3) + 1] = (byte)y;
            }
        }

        return new ImageInput(width, height, rgb);
    }

    [Theory]
    [InlineData(false, 6, 6, 1, 1)]
    [InlineData(true, 6, 6, 2, 2)]
    [InlineData(true, 7, 5, 2, 1)]
    [InlineData(false, 7, 5, 2, 1)]
    [InlineData(true, 9, 9, 3, 3)]
    [InlineData(false, 9, 9, 3, 3)]
    [InlineData(true, 8, 8, 2, 2)]
    public void Offsets_follow_the_rounding(bool nearest, int width, int height, int left, int top)
    {
        var processor = new ImageProcessor(new ImageProcessorConfig
        {
            ResizeMode = ImageResizeMode.None,
            CropWidth = 3,
            CropHeight = 3,
            CropOffsetNearest = nearest,
            Rescale = 1f,
        });
        var processed = processor.Process(Gradient(width, height));
        Assert.Equal(left, processed.Planes[0]);
        Assert.Equal(top, processed.Planes[3 * 3]);
    }

    [Fact]
    public void The_rounding_survives_the_json_round_trip()
    {
        var config = new ImageProcessorConfig { CropWidth = 4, CropHeight = 4, CropOffsetNearest = true };
        using var document = System.Text.Json.JsonDocument.Parse(config.ToJson().ToJsonString());
        Assert.True(ImageProcessorConfig.FromJson(document.RootElement).CropOffsetNearest);
    }
}
