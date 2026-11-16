using Avalonia;

namespace Sharplight.Studio.Controls;

/// <summary>
/// The zoom and pan of an image in a viewport, as pure math: the image is drawn at <see cref="Scale"/> with its top-left corner at
/// <see cref="Offset"/> (viewport pixels). <see cref="Fit"/> centres it at the largest size that shows all of it.
/// </summary>
internal readonly record struct ViewTransform(double Scale, Point Offset)
{
    /// <summary>The smallest zoom, relative to the fit zoom.</summary>
    public const double MinRelativeZoom = 0.5;

    /// <summary>The largest zoom (pixel scale; 32 means 32 screen pixels per image pixel).</summary>
    public const double MaxScale = 32;

    /// <summary>The scale at which the whole image fits (never above 100 % for images smaller than the viewport).</summary>
    /// <param name="image">The image's size in pixels.</param>
    /// <param name="viewport">The viewport's size.</param>
    public static double FitScale(Size image, Size viewport)
    {
        if (image.Width <= 0 || image.Height <= 0 || viewport.Width <= 0 || viewport.Height <= 0)
        {
            return 1;
        }

        return Math.Min(viewport.Width / image.Width, viewport.Height / image.Height);
    }

    /// <summary>The image centred at <paramref name="scale"/>.</summary>
    public static ViewTransform Centered(double scale, Size image, Size viewport)
        => new(scale, new Point((viewport.Width - image.Width * scale) / 2, (viewport.Height - image.Height * scale) / 2));

    /// <summary>The image fitted and centred.</summary>
    public static ViewTransform Fit(Size image, Size viewport) => Centered(FitScale(image, viewport), image, viewport);

    /// <summary>The image at 100 % (one screen pixel per image pixel), centred.</summary>
    public static ViewTransform ActualSize(Size image, Size viewport) => Centered(1, image, viewport);

    /// <summary>Zooms by <paramref name="factor"/> keeping the image point under <paramref name="anchor"/> (viewport pixels) fixed.</summary>
    public ViewTransform ZoomAround(Point anchor, double factor, Size image, Size viewport)
    {
        double fit = FitScale(image, viewport);
        double scale = Math.Clamp(Scale * factor, fit * MinRelativeZoom, Math.Max(MaxScale, fit));
        double ratio = scale / Scale;
        var offset = new Point(anchor.X - (anchor.X - Offset.X) * ratio, anchor.Y - (anchor.Y - Offset.Y) * ratio);
        return new ViewTransform(scale, offset).Clamped(image, viewport);
    }

    /// <summary>Moves the image by <paramref name="delta"/> viewport pixels.</summary>
    public ViewTransform Pan(Vector delta, Size image, Size viewport) => (this with { Offset = Offset + delta }).Clamped(image, viewport);

    /// <summary>
    /// Keeps the image from being dragged away: an image smaller than the viewport stays centred on that axis, a larger one covers
    /// the viewport's edges.
    /// </summary>
    public ViewTransform Clamped(Size image, Size viewport)
    {
        return this with { Offset = new Point(ClampAxis(Offset.X, image.Width * Scale, viewport.Width), ClampAxis(Offset.Y, image.Height * Scale, viewport.Height)) };
    }

    /// <summary>The image point (pixels) under a viewport point.</summary>
    public Point ToImage(Point viewportPoint) => new((viewportPoint.X - Offset.X) / Scale, (viewportPoint.Y - Offset.Y) / Scale);

    private static double ClampAxis(double offset, double extent, double viewport)
        => extent <= viewport ? (viewport - extent) / 2 : Math.Clamp(offset, viewport - extent, 0);
}
