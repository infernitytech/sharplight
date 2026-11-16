using Sharplight.Vision;

namespace Sharplight.Video.Tests;

public sealed class VideoClipTests
{
    [Fact]
    public void ReportsSizeAndDurationOfTheFirstFrame()
    {
        VideoClip clip = TestClips.MovingGradient(16, 8, 12, 4);

        Assert.Equal(16, clip.Width);
        Assert.Equal(8, clip.Height);
        Assert.Equal(TimeSpan.FromSeconds(3), clip.Duration);
    }

    [Fact]
    public void RejectsFramesOfDifferentSizes()
    {
        var frames = new[] { new ImageInput(2, 2, new byte[12]), new ImageInput(3, 2, new byte[18]) };

        Assert.Throws<ArgumentException>(() => new VideoClip(frames, 10));
    }

    [Fact]
    public void RejectsEmptyClipsBadFrameDataAndBadRates()
    {
        Assert.Throws<ArgumentException>(() => new VideoClip([], 10));
        Assert.Throws<ArgumentException>(() => new VideoClip([new ImageInput(2, 2, new byte[5])], 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => new VideoClip([new ImageInput(1, 1, new byte[3])], 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new VideoClip([new ImageInput(1, 1, new byte[3])], double.NaN));
    }
}
