using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace Sharplight.Studio.Controls;

/// <summary>Stacked bars (a value plus an optional secondary value on top) with value grid lines and thinned-out labels.</summary>
internal sealed class BarChart : Control
{
    /// <summary>The bars.</summary>
    public static readonly StyledProperty<IReadOnlyList<ChartBar>?> BarsProperty = AvaloniaProperty.Register<BarChart, IReadOnlyList<ChartBar>?>(nameof(Bars));

    /// <summary>Fill of the main value.</summary>
    public static readonly StyledProperty<IBrush?> FillProperty = AvaloniaProperty.Register<BarChart, IBrush?>(nameof(Fill));

    /// <summary>Fill of the secondary value.</summary>
    public static readonly StyledProperty<IBrush?> SecondaryFillProperty = AvaloniaProperty.Register<BarChart, IBrush?>(nameof(SecondaryFill));

    /// <summary>Colour of grid lines and labels.</summary>
    public static readonly StyledProperty<IBrush?> AxisBrushProperty = AvaloniaProperty.Register<BarChart, IBrush?>(nameof(AxisBrush));

    private const double LabelWidth = 44;
    private const double LabelHeight = 18;

    static BarChart()
    {
        AffectsRender<BarChart>(BarsProperty, FillProperty, SecondaryFillProperty, AxisBrushProperty);
    }

    /// <summary>The bars.</summary>
    public IReadOnlyList<ChartBar>? Bars { get => GetValue(BarsProperty); set => SetValue(BarsProperty, value); }

    /// <summary>Fill of the main value.</summary>
    public IBrush? Fill { get => GetValue(FillProperty); set => SetValue(FillProperty, value); }

    /// <summary>Fill of the secondary value.</summary>
    public IBrush? SecondaryFill { get => GetValue(SecondaryFillProperty); set => SetValue(SecondaryFillProperty, value); }

    /// <summary>Colour of grid lines and labels.</summary>
    public IBrush? AxisBrush { get => GetValue(AxisBrushProperty); set => SetValue(AxisBrushProperty, value); }

    /// <inheritdoc/>
    public override void Render(DrawingContext context)
    {
        var bars = Bars ?? [];
        var axis = AxisBrush ?? Brushes.Gray;
        var plot = new Rect(LabelWidth, 6, Math.Max(1, Bounds.Width - LabelWidth - 8), Math.Max(1, Bounds.Height - LabelHeight - 10));
        double max = LineChart.NiceMax(bars.Count == 0 ? 0 : bars.Max(bar => bar.Value + bar.Secondary));
        for (int i = 0; i <= 4; i++)
        {
            double y = plot.Bottom - (plot.Height * i / 4);
            context.DrawLine(new Pen(axis, i == 0 ? 1 : 0.5), new Point(plot.X, y), new Point(plot.Right, y));
            var label = Text(Compact(max * i / 4), axis);
            context.DrawText(label, new Point(plot.X - label.Width - 6, y - (label.Height / 2)));
        }

        if (bars.Count == 0)
        {
            return;
        }

        double slot = plot.Width / bars.Count;
        double width = Math.Max(2, slot * 0.62);
        var fill = Fill ?? Brushes.MediumPurple;
        var secondary = SecondaryFill ?? Brushes.DeepSkyBlue;
        int every = Math.Max(1, (int)Math.Ceiling(bars.Count / Math.Max(1, plot.Width / 64)));
        for (int i = 0; i < bars.Count; i++)
        {
            var bar = bars[i];
            double x = plot.X + (slot * i) + ((slot - width) / 2);
            double h1 = bar.Value / max * plot.Height;
            double h2 = bar.Secondary / max * plot.Height;
            if (h1 > 0)
            {
                context.DrawRectangle(fill, null, new Rect(x, plot.Bottom - h1, width, h1), 3, 3);
            }

            if (h2 > 0)
            {
                context.DrawRectangle(secondary, null, new Rect(x, plot.Bottom - h1 - h2, width, h2), 3, 3);
            }

            if (i % every == 0)
            {
                var label = Text(bar.Label, axis);
                context.DrawText(label, new Point(x + (width / 2) - (label.Width / 2), plot.Bottom + 4));
            }
        }
    }

    /// <summary><c>1.2k</c>, <c>3.4M</c>.</summary>
    internal static string Compact(double value) => value switch
    {
        >= 1e6 => (value / 1e6).ToString("0.#", CultureInfo.InvariantCulture) + "M",
        >= 1e3 => (value / 1e3).ToString("0.#", CultureInfo.InvariantCulture) + "k",
        _ => value.ToString("0.#", CultureInfo.InvariantCulture),
    };

    private static FormattedText Text(string text, IBrush brush)
        => new(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, Typeface.Default, 10.5, brush);
}
