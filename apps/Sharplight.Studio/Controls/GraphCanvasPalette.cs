using Avalonia.Controls;
using Avalonia.Media;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.Controls;

/// <summary>The theme brushes of a <see cref="GraphCanvas"/>, looked up once per frame.</summary>
internal readonly struct GraphCanvasPalette
{
    private readonly IBrush _linear;
    private readonly IBrush _attention;
    private readonly IBrush _normalization;
    private readonly IBrush _elementwise;
    private readonly IBrush _state;

    public GraphCanvasPalette(GraphCanvas canvas)
    {
        Surface = Find(canvas, "SurfaceBrush", Brushes.White);
        Border = Find(canvas, "BorderSubtleBrush", Brushes.LightGray);
        Text = Find(canvas, "TextPrimaryBrush", Brushes.Black);
        Muted = Find(canvas, "TextMutedBrush", Brushes.Gray);
        Grid = Find(canvas, "BorderSubtleBrush", Brushes.LightGray);
        Accent = Find(canvas, "AccentBrush", Brushes.MediumPurple);
        Secondary = Find(canvas, "AccentSecondaryBrush", Brushes.DeepSkyBlue);
        AccentSoft = Find(canvas, "AccentSoftBrush", Brushes.Lavender);
        var edge = Find(canvas, "GridLineBrush", Brushes.Gray);
        Edge = edge;
        EdgeFaint = edge is ISolidColorBrush solid ? new SolidColorBrush(solid.Color, 0.35) : edge;
        _linear = Accent;
        _attention = Secondary;
        _normalization = Find(canvas, "SuccessBrush", Brushes.SeaGreen);
        _elementwise = Find(canvas, "WarningBrush", Brushes.Orange);
        _state = Find(canvas, "DangerBrush", Brushes.IndianRed);
    }

    public IBrush Surface { get; }

    public IBrush Border { get; }

    public IBrush Text { get; }

    public IBrush Muted { get; }

    public IBrush Grid { get; }

    public IBrush Accent { get; }

    public IBrush Secondary { get; }

    public IBrush AccentSoft { get; }

    public IBrush Edge { get; }

    public IBrush EdgeFaint { get; }

    public IBrush Category(OpCategory category) => category switch
    {
        OpCategory.Linear => _linear,
        OpCategory.Attention => _attention,
        OpCategory.Normalization => _normalization,
        OpCategory.Elementwise => _elementwise,
        OpCategory.State => _state,
        _ => Muted,
    };

    private static IBrush Find(GraphCanvas canvas, string key, IBrush fallback) =>
        canvas.TryFindResource(key, canvas.ActualThemeVariant, out var value) && value is IBrush brush ? brush : fallback;
}
