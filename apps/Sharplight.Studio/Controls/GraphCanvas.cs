using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Media;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.Controls;

/// <summary>
/// Draws a <see cref="GraphLayout"/> as boxes and dataflow curves, with panning (drag), zooming (wheel, around the pointer) and
/// selection (click). Only what is inside the viewport is drawn and text is dropped when zoomed far out, so graphs with 100,000
/// nodes stay responsive. Colors come from the app's theme resources, so it follows light and dark themes and the accent palette.
/// </summary>
internal sealed class GraphCanvas : Control
{
    /// <summary>The layout to draw.</summary>
    public static readonly StyledProperty<GraphLayout?> LayoutProperty = AvaloniaProperty.Register<GraphCanvas, GraphLayout?>(nameof(Layout));

    /// <summary>Id of the selected node, -1 for none; two-way by default.</summary>
    public static readonly StyledProperty<int> SelectedNodeIdProperty =
        AvaloniaProperty.Register<GraphCanvas, int>(nameof(SelectedNodeId), -1, defaultBindingMode: BindingMode.TwoWay);

    /// <summary>Op whose nodes are highlighted, or null.</summary>
    public static readonly StyledProperty<string?> HighlightOpProperty = AvaloniaProperty.Register<GraphCanvas, string?>(nameof(HighlightOp));

    /// <summary>Smallest zoom factor.</summary>
    public const double MinZoom = 0.02;

    /// <summary>Largest zoom factor.</summary>
    public const double MaxZoom = 3;

    private const double DefaultZoom = 0.9;
    private const double TextZoom = 0.42;
    private const double DragThreshold = 4;
    private const double ViewMargin = 32;
    private const int LongEdgeRanks = 6;
    private const int MaxTextBoxes = 2500;
    private const double WheelStep = 60;

    private static readonly FontFamily s_mono = new("Cascadia Code, JetBrains Mono, Consolas, DejaVu Sans Mono, Menlo, monospace");

    private readonly List<(double Y, string Group)> _groupStarts = [];
    private Point _origin;
    private double _zoom = DefaultZoom;
    private bool _resetPending = true;
    private Point? _pressPoint;
    private Point _lastPoint;
    private bool _dragging;
    private bool _selectingByPointer;

    static GraphCanvas()
    {
        AffectsRender<GraphCanvas>(LayoutProperty, SelectedNodeIdProperty, HighlightOpProperty);
        FocusableProperty.OverrideDefaultValue<GraphCanvas>(true);
        ClipToBoundsProperty.OverrideDefaultValue<GraphCanvas>(true);
    }

    /// <summary>Creates the canvas.</summary>
    public GraphCanvas()
    {
        ActualThemeVariantChanged += (_, _) => InvalidateVisual();
        ResourcesChanged += (_, _) => InvalidateVisual();
    }

    /// <summary>The layout to draw.</summary>
    public GraphLayout? Layout { get => GetValue(LayoutProperty); set => SetValue(LayoutProperty, value); }

    /// <summary>Id of the selected node, -1 for none.</summary>
    public int SelectedNodeId { get => GetValue(SelectedNodeIdProperty); set => SetValue(SelectedNodeIdProperty, value); }

    /// <summary>Op whose nodes are highlighted, or null.</summary>
    public string? HighlightOp { get => GetValue(HighlightOpProperty); set => SetValue(HighlightOpProperty, value); }

    /// <summary>The current zoom factor.</summary>
    public double Zoom => _zoom;

    /// <summary>The graph point at the top-left corner of the control.</summary>
    public Point Origin => _origin;

    /// <summary>Zooms by <paramref name="factor"/> around the center of the control.</summary>
    public void ZoomBy(double factor) => ZoomAround(new Point(Bounds.Width / 2, Bounds.Height / 2), factor);

    /// <summary>Shows the whole graph.</summary>
    public void FitAll()
    {
        if (Layout is not { } layout || Bounds.Width <= 0 || Bounds.Height <= 0)
        {
            return;
        }

        double width = layout.Right - layout.Left + (2 * ViewMargin);
        double height = layout.Bottom + (2 * ViewMargin);
        _zoom = Math.Clamp(Math.Min(Bounds.Width / width, Bounds.Height / height), MinZoom, MaxZoom);
        _origin = new Point(((layout.Left + layout.Right) / 2) - (Bounds.Width / 2 / _zoom), -ViewMargin);
        InvalidateVisual();
    }

    /// <summary>Back to the default zoom at the top of the graph.</summary>
    public void ResetView()
    {
        ApplyReset();
        InvalidateVisual();
    }

    /// <summary>Scrolls the selected node to the center.</summary>
    public void CenterOnSelection()
    {
        if (Layout?.NodeOf(SelectedNodeId) is { } box)
        {
            CenterOn(box);
        }
    }

    /// <summary>The box at a point of the control, or null.</summary>
    public LayoutNode? HitTest(Point point)
    {
        if (Layout is not { } layout)
        {
            return null;
        }

        var graph = ToGraph(point);
        foreach (var box in layout.Nodes)
        {
            if (graph.X >= box.X && graph.X <= box.X + box.Width && graph.Y >= box.Y && graph.Y <= box.Y + box.Height)
            {
                return box;
            }
        }

        return null;
    }

    /// <summary>Where a graph point appears in the control.</summary>
    public Point ToScreen(double x, double y) => new((x - _origin.X) * _zoom, (y - _origin.Y) * _zoom);

    /// <inheritdoc/>
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == LayoutProperty)
        {
            BuildGroupStarts();
            _resetPending = true;
        }
        else if (change.Property == SelectedNodeIdProperty && !_selectingByPointer && Layout?.NodeOf(SelectedNodeId) is { } box && !IsOnScreen(box))
        {
            CenterOn(box);
        }
    }

    /// <inheritdoc/>
    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        var point = e.GetCurrentPoint(this);
        if (!point.Properties.IsLeftButtonPressed && !point.Properties.IsMiddleButtonPressed)
        {
            return;
        }

        _pressPoint = point.Position;
        _lastPoint = point.Position;
        _dragging = false;
        e.Pointer.Capture(this);
        Focus();
        e.Handled = true;
    }

    /// <inheritdoc/>
    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        if (_pressPoint is not { } press)
        {
            return;
        }

        var position = e.GetPosition(this);
        if (!_dragging && (Math.Abs(position.X - press.X) > DragThreshold || Math.Abs(position.Y - press.Y) > DragThreshold))
        {
            _dragging = true;
            Cursor = new Cursor(StandardCursorType.SizeAll);
        }

        if (_dragging)
        {
            var delta = position - _lastPoint;
            _origin -= new Vector(delta.X / _zoom, delta.Y / _zoom);
            _lastPoint = position;
            InvalidateVisual();
        }
    }

    /// <inheritdoc/>
    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        if (_pressPoint is null)
        {
            return;
        }

        if (!_dragging)
        {
            var box = HitTest(e.GetPosition(this));
            _selectingByPointer = true;
            SetCurrentValue(SelectedNodeIdProperty, box?.NodeId ?? -1);
            _selectingByPointer = false;
        }

        _pressPoint = null;
        _dragging = false;
        Cursor = null;
        e.Pointer.Capture(null);
        e.Handled = true;
    }

    /// <inheritdoc/>
    protected override void OnPointerWheelChanged(PointerWheelEventArgs e)
    {
        base.OnPointerWheelChanged(e);
        if (e.KeyModifiers.HasFlag(KeyModifiers.Control) || e.KeyModifiers.HasFlag(KeyModifiers.Meta))
        {
            ZoomAround(e.GetPosition(this), Math.Pow(1.15, e.Delta.Y));
        }
        else
        {
            var delta = e.KeyModifiers.HasFlag(KeyModifiers.Shift) ? new Vector(e.Delta.Y, e.Delta.X) : e.Delta;
            _origin -= new Vector(delta.X * WheelStep / _zoom, delta.Y * WheelStep / _zoom);
            InvalidateVisual();
        }

        e.Handled = true;
    }

    /// <inheritdoc/>
    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e);
        double step = 80 / _zoom;
        switch (e.Key)
        {
            case Key.Up: _origin -= new Vector(0, step); break;
            case Key.Down: _origin += new Vector(0, step); break;
            case Key.Left: _origin -= new Vector(step, 0); break;
            case Key.Right: _origin += new Vector(step, 0); break;
            case Key.PageUp: _origin -= new Vector(0, Bounds.Height / _zoom * 0.9); break;
            case Key.PageDown: _origin += new Vector(0, Bounds.Height / _zoom * 0.9); break;
            case Key.Add or Key.OemPlus: ZoomBy(1.25); break;
            case Key.Subtract or Key.OemMinus: ZoomBy(0.8); break;
            default: return;
        }

        InvalidateVisual();
        e.Handled = true;
    }

    /// <inheritdoc/>
    public override void Render(DrawingContext context)
    {
        if (Layout is not { } layout || Bounds.Width <= 0 || Bounds.Height <= 0)
        {
            return;
        }

        if (_resetPending)
        {
            _resetPending = false;
            ApplyReset();
            if (layout.NodeOf(SelectedNodeId) is { } selectedBox)
            {
                ApplyCenter(selectedBox);
            }
        }

        var palette = new GraphCanvasPalette(this);
        context.DrawRectangle(Brushes.Transparent, null, new Rect(Bounds.Size));
        var view = new Rect(_origin.X, _origin.Y, Bounds.Width / _zoom, Bounds.Height / _zoom);
        var visible = view.Inflate(new Thickness(GraphLayout.NodeWidth, GraphLayout.RankPitch));
        int selectedIndex = layout.NodeOf(SelectedNodeId)?.Index ?? -1;

        DrawGroups(context, palette, view);
        DrawEdges(context, layout, palette, visible, selectedIndex);

        int count = 0;
        foreach (var box in layout.Nodes)
        {
            if (box.X + box.Width >= view.X && box.X <= view.Right && box.Y + box.Height >= view.Y && box.Y <= view.Bottom)
            {
                count++;
            }
        }

        bool text = _zoom >= TextZoom && count <= MaxTextBoxes;
        foreach (var box in layout.Nodes)
        {
            if (box.X + box.Width >= view.X && box.X <= view.Right && box.Y + box.Height >= view.Y && box.Y <= view.Bottom)
            {
                DrawBox(context, box, palette, box.Index == selectedIndex, text);
            }
        }
    }

    private void DrawGroups(DrawingContext context, GraphCanvasPalette palette, Rect view)
    {
        if (_zoom < TextZoom * 0.6)
        {
            return;
        }

        var pen = new Pen(palette.Grid, 1, new DashStyle([4, 4], 0));
        foreach (var (y, group) in _groupStarts)
        {
            if (y < view.Y - GraphLayout.RankPitch || y > view.Bottom)
            {
                continue;
            }

            double screenY = (y - _origin.Y) * _zoom;
            context.DrawLine(pen, new Point(0, screenY), new Point(Bounds.Width, screenY));
            var label = Text(group, palette.Muted, 11, FontWeight.Medium, null, double.PositiveInfinity);
            var labelRect = new Rect(8, screenY + 3, label.Width + 10, label.Height + 2);
            context.DrawRectangle(palette.Surface, null, labelRect, 4, 4);
            context.DrawText(label, new Point(13, screenY + 4));
        }
    }

    private void DrawEdges(DrawingContext context, GraphLayout layout, GraphCanvasPalette palette, Rect visible, int selectedIndex)
    {
        var normal = new Pen(palette.Edge, Math.Max(0.6, 1.3 * Math.Sqrt(_zoom)));
        var faint = new Pen(palette.EdgeFaint, Math.Max(0.5, 1.1 * Math.Sqrt(_zoom)));
        var selected = new Pen(palette.Accent, Math.Max(1.2, 2.2 * Math.Sqrt(_zoom)));
        var highlighted = new List<LayoutEdge>();
        foreach (var edge in layout.Edges)
        {
            if (edge.From == selectedIndex || edge.To == selectedIndex)
            {
                highlighted.Add(edge);
                continue;
            }

            var from = layout.Nodes[edge.From];
            var to = layout.Nodes[edge.To];
            if (Math.Max(from.Y, to.Y) < visible.Y || Math.Min(from.Y, to.Y) > visible.Bottom
                || Math.Max(from.CenterX, to.X + to.Width) < visible.X || Math.Min(from.CenterX, to.X) > visible.Right)
            {
                continue;
            }

            DrawEdge(context, from, to, edge.Port, to.Rank - from.Rank > LongEdgeRanks ? faint : normal);
        }

        foreach (var edge in highlighted)
        {
            DrawEdge(context, layout.Nodes[edge.From], layout.Nodes[edge.To], edge.Port, selected);
        }
    }

    private void DrawEdge(DrawingContext context, LayoutNode from, LayoutNode to, double port, Pen pen)
    {
        var start = ToScreen(from.CenterX, from.Y + from.Height);
        var end = ToScreen(to.X + (to.Width * port), to.Y);
        double bend = Math.Max(18 * _zoom, Math.Min(90 * _zoom, (end.Y - start.Y) / 2));
        var geometry = new StreamGeometry();
        using (var g = geometry.Open())
        {
            g.BeginFigure(start, false);
            g.CubicBezierTo(new Point(start.X, start.Y + bend), new Point(end.X, end.Y - bend), end);
            g.EndFigure(false);
        }

        context.DrawGeometry(null, pen, geometry);
        if (_zoom >= TextZoom)
        {
            double r = 2.4 * Math.Sqrt(_zoom);
            context.DrawEllipse(pen.Brush, null, end, r, r);
        }
    }

    private void DrawBox(DrawingContext context, LayoutNode box, GraphCanvasPalette palette, bool selected, bool text)
    {
        var topLeft = ToScreen(box.X, box.Y);
        var rect = new Rect(topLeft, new Size(box.Width * _zoom, box.Height * _zoom));
        bool terminal = box.Kind != LayoutNodeKind.Op;
        double radius = terminal ? rect.Height / 2 : 8 * _zoom;
        bool highlighted = !terminal && HighlightOp is { } op && string.Equals(op, box.Title, StringComparison.Ordinal);
        var border = selected ? new Pen(palette.Accent, Math.Max(1.5, 2 * _zoom))
            : highlighted ? new Pen(palette.Secondary, Math.Max(1.2, 1.8 * _zoom))
            : new Pen(palette.Border, Math.Max(0.6, _zoom));
        context.DrawRectangle(terminal ? palette.AccentSoft : palette.Surface, border, rect, radius, radius);
        if (selected && !terminal)
        {
            context.DrawRectangle(palette.AccentSoft, null, rect, radius, radius);
        }

        if (!terminal)
        {
            var stripe = new Rect(rect.X, rect.Y, Math.Max(2, 4 * _zoom), rect.Height);
            using (context.PushClip(new RoundedRect(rect, radius)))
            {
                context.DrawRectangle(palette.Category(box.Category), null, stripe);
            }
        }

        if (!text)
        {
            return;
        }

        double pad = (terminal ? 14 : 12) * _zoom;
        double width = rect.Width - (2 * pad);
        var title = Text(box.Title, terminal ? palette.Accent : palette.Text, 12.5 * _zoom, FontWeight.SemiBold, null, width);
        var subtitle = Text(box.Subtitle, palette.Muted, 10.5 * _zoom, FontWeight.Normal, s_mono, width);
        if (terminal)
        {
            double total = title.Width + (6 * _zoom) + subtitle.Width;
            if (total <= width)
            {
                double x = rect.X + ((rect.Width - total) / 2);
                context.DrawText(title, new Point(x, rect.Y + ((rect.Height - title.Height) / 2)));
                context.DrawText(subtitle, new Point(x + title.Width + (6 * _zoom), rect.Y + ((rect.Height - subtitle.Height) / 2)));
            }
            else
            {
                context.DrawText(title, new Point(rect.X + pad, rect.Y + ((rect.Height - title.Height) / 2)));
            }

            return;
        }

        double top = rect.Y + ((rect.Height - title.Height - subtitle.Height) / 2);
        context.DrawText(title, new Point(rect.X + pad, top));
        context.DrawText(subtitle, new Point(rect.X + pad, top + title.Height));
        if (box.WeightCount > 0)
        {
            var weights = Text(box.WeightCount == 1 ? "W" : $"W{box.WeightCount}", palette.Muted, 10 * _zoom, FontWeight.SemiBold, null, double.PositiveInfinity);
            context.DrawText(weights, new Point(rect.Right - weights.Width - (8 * _zoom), rect.Y + (5 * _zoom)));
        }
    }

    private FormattedText Text(string text, IBrush brush, double size, FontWeight weight, FontFamily? family, double maxWidth)
    {
        var typeface = new Typeface(family ?? GetValue(TextElement.FontFamilyProperty), FontStyle.Normal, weight);
        var formatted = new FormattedText(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, Math.Max(1, size), brush)
        {
            MaxLineCount = 1,
            Trimming = TextTrimming.CharacterEllipsis,
        };
        if (!double.IsInfinity(maxWidth))
        {
            formatted.MaxTextWidth = Math.Max(1, maxWidth);
        }

        return formatted;
    }

    private void ZoomAround(Point anchor, double factor)
    {
        double zoom = Math.Clamp(_zoom * factor, MinZoom, MaxZoom);
        var graph = ToGraph(anchor);
        _zoom = zoom;
        _origin = new Point(graph.X - (anchor.X / zoom), graph.Y - (anchor.Y / zoom));
        InvalidateVisual();
    }

    private Point ToGraph(Point screen) => new(_origin.X + (screen.X / _zoom), _origin.Y + (screen.Y / _zoom));

    private bool IsOnScreen(LayoutNode box)
    {
        var view = new Rect(_origin.X, _origin.Y, Bounds.Width / _zoom, Bounds.Height / _zoom);
        return Bounds.Width > 0 && view.Contains(new Rect(box.X, box.Y, box.Width, box.Height));
    }

    private void CenterOn(LayoutNode box)
    {
        ApplyCenter(box);
        InvalidateVisual();
    }

    private void ApplyCenter(LayoutNode box)
    {
        if (Bounds.Width <= 0)
        {
            return;
        }

        _zoom = Math.Max(_zoom, TextZoom + 0.18);
        _origin = new Point(box.CenterX - (Bounds.Width / 2 / _zoom), box.CenterY - (Bounds.Height / 2 / _zoom));
    }

    private void ApplyReset()
    {
        if (Layout is not { } layout)
        {
            return;
        }

        _zoom = DefaultZoom;
        double center = layout.Nodes.Count == 0 ? 0 : layout.Nodes.Take(64).Average(n => n.CenterX);
        _origin = new Point(center - (Bounds.Width / 2 / _zoom), -ViewMargin);
    }

    private void BuildGroupStarts()
    {
        _groupStarts.Clear();
        if (Layout is not { } layout)
        {
            return;
        }

        string previous = "";
        foreach (var box in layout.Nodes.Where(b => b.Kind == LayoutNodeKind.Op).OrderBy(b => b.Rank).ThenBy(b => b.Index))
        {
            if (box.Group.Length > 0 && !string.Equals(box.Group, previous, StringComparison.Ordinal))
            {
                _groupStarts.Add((box.Y - ((GraphLayout.RankPitch - GraphLayout.NodeHeight) / 2), box.Group));
                previous = box.Group;
            }
        }
    }
}
