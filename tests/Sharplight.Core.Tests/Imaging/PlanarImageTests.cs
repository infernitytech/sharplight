using Sharplight.Imaging;
using Sharplight.Vision;
using Xunit;

namespace Sharplight.Core.Tests.Imaging;

public class PlanarImageTests
{
    [Theory]
    [InlineData(-1f, 1f)]
    [InlineData(0f, 1f)]
    [InlineData(0f, 255f)]
    public void InterleavedToPlanarAndBackIsLossless(float low, float high)
    {
        var image = PngCodecTests.RandomImage(13, 7, seed: 5);
        var planes = new float[3 * 13 * 7];
        PlanarImage.ToPlanar(image, planes, low, high);
        Assert.All(planes, v => Assert.InRange(v, low, high));
        var back = PlanarImage.FromPlanar(planes, 13, 7, low, high);
        Assert.Equal(image.Rgb, back.Rgb);
    }

    [Fact]
    public void PlanesAreChannelMajor()
    {
        var image = new ImageInput(2, 1, [255, 0, 0, 0, 0, 255]);
        var planes = new float[6];
        PlanarImage.ToPlanar(image, planes);
        Assert.Equal([1f, -1f, -1f, -1f, -1f, 1f], planes);
    }

    [Fact]
    public void FromPlanarClampsAndRounds()
    {
        float[] planes = [-3f, 2f, 0f, 0.002f, 1f, -1f];
        var image = PlanarImage.FromPlanar(planes, 2, 1);
        Assert.Equal(new byte[] { 0, 128, 255, 255, 128, 0 }, image.Rgb);
    }

    [Fact]
    public void RejectsMismatchedSizes()
    {
        Assert.Throws<ArgumentException>(() => PlanarImage.FromPlanar(new float[5], 1, 2));
        Assert.Throws<ArgumentException>(() => PlanarImage.FromPlanar(new float[0], 0, 0));
        Assert.Throws<ArgumentException>(() => PlanarImage.ToPlanar(new ImageInput(1, 1, [1, 2, 3]), new float[2]));
    }
}
