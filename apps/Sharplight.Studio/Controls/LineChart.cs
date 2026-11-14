using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace Sharplight.Studio.Controls;

/// <summary>
/// A light-weight line chart: one or two series, an area gradient under the first, a few horizontal grid lines with value labels
/// and, for time series, start and end time labels. Draws nothing but a hint when there are no points.
/// </summary>
internal sealed class LineChart : Control
{
    /// <summary>The first (main) series.</summary>
    public static readonly StyledProperty<IReadOnlyList<ChartPoint>?> PointsProperty = AvaloniaProperty.Register<LineChart, IReadOnlyList<ChartPoint>?>(nameof(Points));

    /// <summary>An optional second series, drawn with <see cref="SecondaryStroke"/>.</summary>
    public static readonly StyledProperty<IReadOnlyList<ChartPoint>?> SecondaryPointsProperty = AvaloniaProperty.Register<LineChart, IReadOnlyList<ChartPoint>?>(nameof(SecondaryPoints));

    /// <summary>Colour of the main line.</summary>
    public static readonly StyledProperty<IBrush?> StrokeProperty = AvaloniaProperty.Register<LineChart, IBrush?>(nameof(Stroke));

    /// <summary>Colour of the second line.</summary>
    public static readonly StyledProperty<IBrush?> SecondaryStrokeProperty = AvaloniaProperty.Register<LineChart, IBrush?>(nameof(SecondaryStroke));

    /// <summary>Colour of grid lines and labels.</summary>
    public static readonly StyledProperty<IBrush?> AxisBrushProperty = AvaloniaProperty.Register<LineChart, IBrush?>(nameof(AxisBrush));

    /// <summary>Draw grid lines and labels (off for sparklines).</summary>
    public static readonly StyledProperty<bool> ShowAxisProperty = AvaloniaProperty.Register<LineChart, bool>(nameof(ShowAxis), true);

    /// <summary>X values are Unix seconds (labels show times).</summary>
    public static readonly StyledProperty<bool> IsTimeSeriesProperty = AvaloniaProperty.Register<LineChart, bool>(nameof(IsTimeSeries));

    /// <summary>Numeric format of value labels.</summary>
    public static readonly StyledProperty<string> ValueFormatProperty = AvaloniaProperty.Register<LineChart, string>(nameof(ValueFormat), "0.#");

    /// <summary>Fixed top of the value axis (e.g. 100 for percentages); 0 scales to the data.</summary>
    public static readonly StyledProperty<double> MaximumProperty = AvaloniaProperty.Register<LineChart, double>(nameof(Maximum));

    /// <summary>Text drawn when there is no data.</summary>
    public static readonly StyledProperty<string> EmptyTextProperty = AvaloniaProperty.Register<LineChart, string>(nameof(EmptyText), "No data yet");

    private const double LabelWidth = 44;
    private const double LabelHeight = 18;

    static LineChart()
    {
        AffectsRender<LineChart>(PointsProperty, SecondaryPointsProperty, StrokeProperty, SecondaryStrokeProperty, AxisBrushProperty, ShowAxisProperty,
            IsTimeSeriesProperty, ValueFormatProperty, MaximumProperty, EmptyTextProperty);
    }

    /// <summary>The first (main) series.</summary>
    public IReadOnlyList<ChartPoint>? Points { get => GetValue(PointsProperty); set => SetValue(PointsProperty, value); }

    /// <summary>An optional second series.</summary>
    public IReadOnlyList<ChartPoint>? SecondaryPoints { get => GetValue(SecondaryPointsProperty); set => SetValue(SecondaryPointsProperty, value); }

    /// <summary>Colour of the main line.</summary>
    public IBrush? Stroke { get => GetValue(StrokeProperty); set => SetValue(StrokeProperty, value); }

    /// <summary>Colour of the second line.</summary>
    public IBrush? SecondaryStroke { get => GetValue(SecondaryStrokeProperty); set => SetValue(SecondaryStrokeProperty, value); }

    /// <summary>Colour of grid lines and labels.</summary>
    public IBrush? AxisBrush { get => GetValue(AxisBrushProperty); set => SetValue(AxisBrushProperty, value); }

    /// <summary>Draw grid lines and labels.</summary>
    public bool ShowAxis { get => GetValue(ShowAxisProperty); set => SetValue(ShowAxisProperty, value); }

    /// <summary>X values are Unix seconds.</summary>
    public bool IsTimeSeries { get => GetValue(IsTimeSeriesProperty); set => SetValue(IsTimeSeriesProperty, value); }

    /// <summary>Numeric format of value labels.</summary>
    public string ValueFormat { get => GetValue(ValueFormatProperty); set => SetValue(ValueFormatProperty, value); }

    /// <summary>Fixed top of the value axis; 0 scales to the data.</summary>
    public double Maximum { get => GetValue(MaximumProperty); set => SetValue(MaximumProperty, value); }

    /// <summary>Text drawn when there is no data.</summary>
    public string EmptyText { get => GetValue(EmptyTextProperty); set => SetValue(EmptyTextProperty, value); }

    /// <inheritdoc/>
    public override void Render(DrawingContext context)
    {
        var bounds = new Rect(Bounds.Size);
        var axis = AxisBrush ?? Brushes.Gray;
        var main = Points ?? [];
        var second = SecondaryPoints ?? [];
        if (main.Count == 0 && second.Count == 0)
        {
            if (ShowAxis)
            {
                var hint = Text(EmptyText, axis, 12);
                context.DrawText(hint, new Point((bounds.Width - hint.Width) / 2, (bounds.Height - hint.Height) / 2));
            }

            return;
        }

        var all = main.Concat(second).ToList();
        double minX = all.Min(point => point.X);
        double maxX = all.Max(point => point.X);
        double maxY = Maximum > 0 ? Maximum : NiceMax(all.Max(point => point.Y));
        if (maxX <= minX)
        {
            maxX = minX + 1;
        }

        var plot = ShowAxis
            ? new Rect(LabelWidth, 6, Math.Max(1, bounds.Width - LabelWidth - 8), Math.Max(1, bounds.Height - LabelHeight - 10))
            : bounds.Deflate(2);
        if (ShowAxis)
        {
            DrawGrid(context, plot, maxY, axis);
            if (IsTimeSeries)
            {
                DrawTimeLabels(context, plot, minX, maxX, axis);
            }
        }

        Point Map(ChartPoint point) => new(plot.X + ((point.X - minX) / (maxX - minX) * plot.Width), plot.Bottom - (Math.Clamp(point.Y / maxY, 0, 1) * plot.Height));

        DrawSeries(context, main, Map, Stroke ?? Brushes.MediumPurple, plot, fill: true);
        DrawSeries(context, second, Map, SecondaryStroke ?? Brushes.DeepSkyBlue, plot, fill: false);
    }

    /// <summary>A round number at or above <paramref name="value"/> (1, 2, 2.5, 5 times a power of ten).</summary>
    internal static double NiceMax(double value)
    {
        if (value <= 0 || double.IsNaN(value))
        {
            return 1;
        }

        double power = Math.Pow(10, Math.Floor(Math.Log10(value)));
        foreach (double step in new[] { 1, 2, 2.5, 5, 10 })
        {
            if (step * power >= value)
            {
                return step * power;
            }
        }

        return 10 * power;
    }

    private void DrawGrid(DrawingContext context, Rect plot, double maxY, IBrush axis)
    {
        for (int i = 0; i <= 4; i++)
        {
            double y = plot.Bottom - (plot.Height * i / 4);
            context.DrawLine(new Pen(axis, i == 0 ? 1 : 0.5), new Point(plot.X, y), new Point(plot.Right, y));
            var label = Text((maxY * i / 4).ToString(ValueFormat, CultureInfo.InvariantCulture), axis, 10.5);
            context.DrawText(label, new Point(plot.X - label.Width - 6, y - (label.Height / 2)));
        }
    }

    private static void DrawTimeLabels(DrawingContext context, Rect plot, double minX, double maxX, IBrush axis)
    {
        var span = TimeSpan.FromSeconds(maxX - minX);
        string format = span > TimeSpan.FromDays(2) ? "dd MMM" : "HH:mm";
        var start = Text(DateTimeOffset.FromUnixTimeSeconds((long)minX).ToLocalTime().ToString(format, CultureInfo.InvariantCulture), axis, 10.5);
        var end = Text(DateTimeOffset.FromUnixTimeSeconds((long)maxX).ToLocalTime().ToString(format, CultureInfo.InvariantCulture), axis, 10.5);
        context.DrawText(start, new Point(plot.X, plot.Bottom + 4));
        context.DrawText(end, new Point(plot.Right - end.Width, plot.Bottom + 4));
    }

    private static void DrawSeries(DrawingContext context, IReadOnlyList<ChartPoint> points, Func<ChartPoint, Point> map, IBrush stroke, Rect plot, bool fill)
    {
        if (points.Count == 0)
        {
            return;
        }

        var ordered = points.OrderBy(point => point.X).Select(map).ToList();
        var color = stroke is ISolidColorBrush solid ? solid.Color : Colors.MediumPurple;
        if (ordered.Count == 1)
        {
            context.DrawEllipse(stroke, null, ordered[0], 3, 3);
            return;
        }

        if (fill)
        {
            var area = new StreamGeometry();
            using (var g = area.Open())
            {
                g.BeginFigure(new Point(ordered[0].X, plot.Bottom), true);
                foreach (var point in ordered)
                {
                    g.LineTo(point);
                }

                g.LineTo(new Point(ordered[^1].X, plot.Bottom));
                g.EndFigure(true);
            }

            var gradient = new LinearGradientBrush
            {
                StartPoint = new RelativePoint(0, 0, RelativeUnit.Relative),
                EndPoint = new RelativePoint(0, 1, RelativeUnit.Relative),
                GradientStops = [new GradientStop(Color.FromArgb(90, color.R, color.G, color.B), 0), new GradientStop(Color.FromArgb(0, color.R, color.G, color.B), 1)],
            };
            context.DrawGeometry(gradient, null, area);
        }

        var line = new StreamGeometry();
        using (var g = line.Open())
        {
            g.BeginFigure(ordered[0], false);
            foreach (var point in ordered.Skip(1))
            {
                g.LineTo(point);
            }

            g.EndFigure(false);
        }

        context.DrawGeometry(null, new Pen(stroke, 2, lineCap: PenLineCap.Round, lineJoin: PenLineJoin.Round), line);
        if (ordered.Count <= 40)
        {
            foreach (var point in ordered)
            {
                context.DrawEllipse(stroke, null, point, 2.5, 2.5);
            }
        }
    }

    private static FormattedText Text(string text, IBrush brush, double size)
        => new(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Typeface.Default, size, brush);
}
