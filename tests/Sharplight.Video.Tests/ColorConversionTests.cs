using Sharplight.Vision;

namespace Sharplight.Video.Tests;

public sealed class ColorConversionTests
{
    [Theory]
    [InlineData(0, 0, 0, 16, 128, 128)]
    [InlineData(255, 255, 255, 235, 128, 128)]
    [InlineData(255, 0, 0, 82, 90, 240)]
    [InlineData(0, 0, 255, 41, 240, 110)]
    public void UsesBt601LimitedRange(int r, int g, int b, int y, int u, int v)
    {
        (byte[] yp, byte[] up, byte[] vp) = RgbToI420.AllocatePlanes(2, 2);
        var rgb = new byte[12];
        for (int i = 0; i < 4; i++)
        {
            rgb[i * 3] = (byte)r;
            rgb[i * 3 + 1] = (byte)g;
            rgb[i * 3 + 2] = (byte)b;
        }

        RgbToI420.Convert(new ImageInput(2, 2, rgb), yp, up, vp);

        Assert.All(yp, value => Assert.InRange(value, y - 1, y + 1));
        Assert.InRange(up[0], u - 1, u + 1);
        Assert.InRange(vp[0], v - 1, v + 1);
    }

    [Fact]
    public void OddSizesGetRoundedUpChromaPlanes()
    {
        (byte[] y, byte[] u, byte[] v) = RgbToI420.AllocatePlanes(5, 3);

        Assert.Equal((15, 6, 6), (y.Length, u.Length, v.Length));
        RgbToI420.Convert(new ImageInput(5, 3, new byte[45]), y, u, v);
        Assert.All(u, value => Assert.Equal(128, value));
    }
}
