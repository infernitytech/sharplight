using Avalonia;
using Sharplight.Studio.Controls;

namespace Sharplight.Studio.Tests;

/// <summary>The zoom and pan math of the image viewer.</summary>
public sealed class ViewTransformTests
{
    private static readonly Size s_image = new(2000, 1000);
    private static readonly Size s_viewport = new(500, 400);

    [Fact]
    public void Fit_ShowsTheWholeImageCentered()
    {
        var fit = ViewTransform.Fit(s_image, s_viewport);
        Assert.Equal(0.25, fit.Scale);
        Assert.Equal(new Point(0, 75), fit.Offset);
    }

    [Fact]
    public void ActualSize_IsCenteredAtOne()
    {
        var actual = ViewTransform.ActualSize(s_image, s_viewport);
        Assert.Equal(1, actual.Scale);
        Assert.Equal(new Point(-750, -300), actual.Offset);
    }

    [Fact]
    public void ZoomAround_KeepsThePointUnderTheCursorFixed()
    {
        var start = ViewTransform.Fit(s_image, s_viewport);
        var cursor = new Point(130, 160);
        var before = start.ToImage(cursor);
        var zoomed = start.ZoomAround(cursor, 4, s_image, s_viewport);
        Assert.Equal(1, zoomed.Scale, 6);
        var after = zoomed.ToImage(cursor);
        Assert.Equal(before.X, after.X, 6);
        Assert.Equal(before.Y, after.Y, 6);
    }

    [Fact]
    public void Zoom_IsLimited()
    {
        var fit = ViewTransform.Fit(s_image, s_viewport);
        Assert.Equal(fit.Scale * ViewTransform.MinRelativeZoom, fit.ZoomAround(default, 0.0001, s_image, s_viewport).Scale, 9);
        Assert.Equal(ViewTransform.MaxScale, fit.ZoomAround(default, 100000, s_image, s_viewport).Scale, 9);
    }

    [Fact]
    public void Pan_CannotLeaveTheImage_AndSmallImagesStayCentered()
    {
        var zoomed = ViewTransform.ActualSize(s_image, s_viewport).Pan(new Vector(100000, 100000), s_image, s_viewport);
        Assert.Equal(new Point(0, 0), zoomed.Offset);
        var far = zoomed.Pan(new Vector(-100000, -100000), s_image, s_viewport);
        Assert.Equal(new Point(500 - 2000, 400 - 1000), far.Offset);

        var small = ViewTransform.Fit(s_image, s_viewport).Pan(new Vector(90, 90), s_image, s_viewport);
        Assert.Equal(new Point(0, 75), small.Offset);
    }
}
