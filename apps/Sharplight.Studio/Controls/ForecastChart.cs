using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace Sharplight.Studio.Controls;

/// <summary>
/// The chart of the Time series page: the history as a line, the forecast as a dashed line continuing it and a shaded band between a lower and an upper
/// series (the prediction interval). Unlike <see cref="LineChart"/> the value axis fits the data, negative values included; a point whose value is
/// NaN is a gap. X values are positions (history first, then the forecast steps).
/// </summary>
internal sealed class ForecastChart : Control
{
    /// <summary>The observed values.</summary>
    public static readonly StyledProperty<IReadOnlyList<ChartPoint>?> HistoryProperty = AvaloniaProperty.Register<ForecastChart, IReadOnlyList<ChartPoint>?>(nameof(History));

    /// <summary>The point forecast.</summary>
    public static readonly StyledProperty<IReadOnlyList<ChartPoint>?> ForecastProperty = AvaloniaProperty.Register<ForecastChart, IReadOnlyList<ChartPoint>?>(nameof(Forecast));

    /// <summary>The lower edge of the band (same x values as <see cref="Upper"/>).</summary>
    public static readonly StyledProperty<IReadOnlyList<ChartPoint>?> LowerProperty = AvaloniaProperty.Register<ForecastChart, IReadOnlyList<ChartPoint>?>(nameof(Lower));

    /// <summary>The upper edge of the band.</summary>
    public static readonly StyledProperty<IReadOnlyList<ChartPoint>?> UpperProperty = AvaloniaProperty.Register<ForecastChart, IReadOnlyList<ChartPoint>?>(nameof(Upper));

    /// <summary>Colour of the history line.</summary>
    public static readonly StyledProperty<IBrush?> HistoryStrokeProperty = AvaloniaProperty.Register<ForecastChart, IBrush?>(nameof(HistoryStroke));

    /// <summary>Colour of the forecast line and the band.</summary>
    public static readonly StyledProperty<IBrush?> ForecastStrokeProperty = AvaloniaProperty.Register<ForecastChart, IBrush?>(nameof(ForecastStroke));

    /// <summary>Colour of grid lines and labels.</summary>
    public static readonly StyledProperty<IBrush?> AxisBrushProperty = AvaloniaProperty.Register<ForecastChart, IBrush?>(nameof(AxisBrush));

    /// <summary>The label under the left end of the x axis.</summary>
    public static readonly StyledProperty<string> StartLabelProperty = AvaloniaProperty.Register<ForecastChart, string>(nameof(StartLabel), "");

    /// <summary>The label under the right end of the x axis.</summary>
    public static readonly StyledProperty<string> EndLabelProperty = AvaloniaProperty.Register<ForecastChart, string>(nameof(EndLabel), "");

    /// <summary>Text drawn when there is no data.</summary>
    public static readonly StyledProperty<string> EmptyTextProperty = AvaloniaProperty.Register<ForecastChart, string>(nameof(EmptyText), "No forecast yet");

    private const double LabelWidth = 52;
    private const double LabelHeight = 18;

    static ForecastChart()
    {
        AffectsRender<ForecastChart>(HistoryProperty, ForecastProperty, LowerProperty, UpperProperty, HistoryStrokeProperty, ForecastStrokeProperty, AxisBrushProperty,
            StartLabelProperty, EndLabelProperty, EmptyTextProperty);
    }

    /// <summary>The observed values.</summary>
    public IReadOnlyList<ChartPoint>? History { get => GetValue(HistoryProperty); set => SetValue(HistoryProperty, value); }

    /// <summary>The point forecast.</summary>
    public IReadOnlyList<ChartPoint>? Forecast { get => GetValue(ForecastProperty); set => SetValue(ForecastProperty, value); }

    /// <summary>The lower edge of the band.</summary>
    public IReadOnlyList<ChartPoint>? Lower { get => GetValue(LowerProperty); set => SetValue(LowerProperty, value); }

    /// <summary>The upper edge of the band.</summary>
    public IReadOnlyList<ChartPoint>? Upper { get => GetValue(UpperProperty); set => SetValue(UpperProperty, value); }

    /// <summary>Colour of the history line.</summary>
    public IBrush? HistoryStroke { get => GetValue(HistoryStrokeProperty); set => SetValue(HistoryStrokeProperty, value); }

    /// <summary>Colour of the forecast line and the band.</summary>
    public IBrush? ForecastStroke { get => GetValue(ForecastStrokeProperty); set => SetValue(ForecastStrokeProperty, value); }

    /// <summary>Colour of grid lines and labels.</summary>
    public IBrush? AxisBrush { get => GetValue(AxisBrushProperty); set => SetValue(AxisBrushProperty, value); }

    /// <summary>The label under the left end of the x axis.</summary>
    public string StartLabel { get => GetValue(StartLabelProperty); set => SetValue(StartLabelProperty, value); }

    /// <summary>The label under the right end of the x axis.</summary>
    public string EndLabel { get => GetValue(EndLabelProperty); set => SetValue(EndLabelProperty, value); }

    /// <summary>Text drawn when there is no data.</summary>
    public string EmptyText { get => GetValue(EmptyTextProperty); set => SetValue(EmptyTextProperty, value); }

    /// <inheritdoc/>
    public override void Render(DrawingContext context)
    {
        var bounds = new Rect(Bounds.Size);
        var axis = AxisBrush ?? Brushes.Gray;
        var history = Finite(History);
        var forecast = Finite(Forecast);
        var lower = Finite(Lower);
        var upper = Finite(Upper);
        var all = history.Concat(forecast).Concat(lower).Concat(upper).ToList();
        if (all.Count == 0)
        {
            var hint = Text(EmptyText, axis, 12);
            context.DrawText(hint, new Point((bounds.Width - hint.Width) / 2, (bounds.Height - hint.Height) / 2));
            return;
        }

        var (minY, maxY) = ValueRange(all.Select(point => point.Y));
        double minX = all.Min(point => point.X);
        double maxX = Math.Max(all.Max(point => point.X), minX + 1);
        var plot = new Rect(LabelWidth, 6, Math.Max(1, bounds.Width - LabelWidth - 8), Math.Max(1, bounds.Height - LabelHeight - 10));
        DrawGrid(context, plot, minY, maxY, axis);
        DrawLabels(context, plot, axis);

        Point Map(ChartPoint point) => new(plot.X + ((point.X - minX) / (maxX - minX) * plot.Width), plot.Bottom - ((point.Y - minY) / (maxY - minY) * plot.Height));

        var forecastBrush = ForecastStroke ?? Brushes.DeepSkyBlue;
        var color = forecastBrush is ISolidColorBrush solid ? solid.Color : Colors.DeepSkyBlue;
        if (forecast.Count > 0)
        {
            double divider = Map(new ChartPoint(forecast.Min(point => point.X), 0)).X;
            context.DrawLine(new Pen(axis, 0.5, new DashStyle([3, 3], 0)), new Point(divider, plot.Top), new Point(divider, plot.Bottom));
        }

        DrawBand(context, lower, upper, Map, Color.FromArgb(60, color.R, color.G, color.B));
        DrawLine(context, history, Map, new Pen(HistoryStroke ?? Brushes.MediumPurple, 2, lineCap: PenLineCap.Round, lineJoin: PenLineJoin.Round));
        DrawLine(context, forecast, Map, new Pen(forecastBrush, 2, new DashStyle([5, 3], 0), PenLineCap.Round, PenLineJoin.Round));
    }

    /// <summary>The value axis range for <paramref name="values"/>: the data's span widened by a tenth, and widened to a unit span for a flat series.</summary>
    /// <param name="values">The finite values; at least one.</param>
    internal static (double Min, double Max) ValueRange(IEnumerable<double> values)
    {
        double min = double.PositiveInfinity;
        double max = double.NegativeInfinity;
        foreach (double value in values)
        {
            min = Math.Min(min, value);
            max = Math.Max(max, value);
        }

        double span = max - min;
        if (span <= 0)
        {
            span = Math.Max(1, Math.Abs(max) * 0.1);
            return (min - (span / 2), max + (span / 2));
        }

        return (min - (span * 0.1), max + (span * 0.1));
    }

    private static List<ChartPoint> Finite(IReadOnlyList<ChartPoint>? points)
        => points is null ? [] : [.. points.Where(point => double.IsFinite(point.X) && double.IsFinite(point.Y))];

    private static void DrawGrid(DrawingContext context, Rect plot, double minY, double maxY, IBrush axis)
    {
        for (int i = 0; i <= 4; i++)
        {
            double y = plot.Bottom - (plot.Height * i / 4);
            context.DrawLine(new Pen(axis, i == 0 ? 1 : 0.5), new Point(plot.X, y), new Point(plot.Right, y));
            var label = Text((minY + ((maxY - minY) * i / 4)).ToString("G4", CultureInfo.InvariantCulture), axis, 10.5);
            context.DrawText(label, new Point(plot.X - label.Width - 6, y - (label.Height / 2)));
        }
    }

    private void DrawLabels(DrawingContext context, Rect plot, IBrush axis)
    {
        if (StartLabel.Length > 0)
        {
            context.DrawText(Text(StartLabel, axis, 10.5), new Point(plot.X, plot.Bottom + 4));
        }

        if (EndLabel.Length > 0)
        {
            var end = Text(EndLabel, axis, 10.5);
            context.DrawText(end, new Point(plot.Right - end.Width, plot.Bottom + 4));
        }
    }

    private static void DrawBand(DrawingContext context, List<ChartPoint> lower, List<ChartPoint> upper, Func<ChartPoint, Point> map, Color fill)
    {
        if (lower.Count < 2 || upper.Count < 2)
        {
            return;
        }

        var band = new StreamGeometry();
        using (var g = band.Open())
        {
            g.BeginFigure(map(upper[0]), true);
            foreach (var point in upper.Skip(1))
            {
                g.LineTo(map(point));
            }

            foreach (var point in Enumerable.Reverse(lower))
            {
                g.LineTo(map(point));
            }

            g.EndFigure(true);
        }

        context.DrawGeometry(new SolidColorBrush(fill), null, band);
    }

    /// <summary>Draws the points as a line; the points are already free of gaps, a jump in x of more than one position starts a new segment.</summary>
    private static void DrawLine(DrawingContext context, List<ChartPoint> points, Func<ChartPoint, Point> map, Pen pen)
    {
        if (points.Count == 0)
        {
            return;
        }

        if (points.Count == 1)
        {
            context.DrawEllipse(pen.Brush, null, map(points[0]), 3, 3);
            return;
        }

        var line = new StreamGeometry();
        using (var g = line.Open())
        {
            g.BeginFigure(map(points[0]), false);
            for (int i = 1; i < points.Count; i++)
            {
                if (points[i].X - points[i - 1].X > 1.5)
                {
                    g.EndFigure(false);
                    g.BeginFigure(map(points[i]), false);
                }
                else
                {
                    g.LineTo(map(points[i]));
                }
            }

            g.EndFigure(false);
        }

        context.DrawGeometry(null, pen, line);
    }

    private static FormattedText Text(string text, IBrush brush, double size)
        => new(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Typeface.Default, size, brush);
}
