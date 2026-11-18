using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Presenters;
using Avalonia.Controls.Primitives;
using Avalonia.VisualTree;
using Sharplight.Studio.Controls;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Walks the visual tree of a window and finds layout that paints over its neighbours: a visual whose visible area (after the clipping of
/// its ancestors) leaves the bounds of its parent, and siblings of a stacking panel that overlap. Scroll content and popups are fine:
/// scroll viewers clip, and popups are not part of the window's visual tree.
/// </summary>
internal static class LayoutChecker
{
    private const double Tolerance = 1.5;

    /// <summary>A wrapping text block narrower than this many characters (of its font size) is squeezed: the words fall one per line.</summary>
    internal const double MinimumWrapCharacters = 8;

    /// <summary>Returns a description of every visual below <paramref name="root"/> that is arranged outside its parent or overlaps a sibling.</summary>
    public static List<string> Find(Visual root, Func<Visual, bool>? focus = null)
    {
        var problems = new List<string>();
        Visit(root, root, problems, focus ?? (_ => true));
        return problems;
    }

    /// <summary>Pictures, viewers, players and cards: what must never be laid out over its surroundings.</summary>
    public static bool IsContentVisual(Visual visual)
        => visual is Image or ImageViewer or ZoomPanView or VideoView || visual is Border border && (border.Classes.Contains("card") || border.Classes.Contains("inset"));

    /// <summary>The bounds of <paramref name="visual"/> in the coordinates of <paramref name="root"/>.</summary>
    public static Rect BoundsIn(Visual visual, Visual root)
    {
        var origin = visual.TranslatePoint(default, root);
        return origin is { } point ? new Rect(point, visual.Bounds.Size) : default;
    }

    /// <summary>The part of <paramref name="visual"/> that is painted: its bounds cut by every clipping ancestor up to and including <paramref name="until"/>.</summary>
    public static Rect Painted(Visual visual, Visual root, Visual until)
    {
        var rect = BoundsIn(visual, root);
        for (var ancestor = visual.GetVisualParent(); ancestor is not null; ancestor = ancestor.GetVisualParent())
        {
            if (Clips(ancestor))
            {
                rect = rect.Intersect(BoundsIn(ancestor, root));
            }

            if (ancestor == until)
            {
                break;
            }
        }

        return rect;
    }

    private static bool Clips(Visual visual) => visual.ClipToBounds || visual is ScrollContentPresenter;

    private static void Visit(Visual visual, Visual root, List<string> problems, Func<Visual, bool> focus)
    {
        // The inside of a control template (a slider's track, a combo box's popup host) is the control's own business.
        if (visual is Slider or ComboBox or ScrollBar or Popup or ToggleSwitch or NumericUpDown or ToggleButton)
        {
            return;
        }

        var children = visual.GetVisualChildren().OfType<Visual>().Where(child => child.IsVisible && child.Bounds.Width > 0 && child.Bounds.Height > 0).ToList();
        var parentRect = BoundsIn(visual, root);
        foreach (var child in children)
        {
            if (child is TextBlock text && FindExtremeWrap(text) is { } wrap)
            {
                problems.Add($"{Describe(child)} wraps in {wrap}");
            }

            var painted = Painted(child, root, visual);
            // Cut by the clips above the parent too: what a clipped ancestor hides cannot overlap anything.
            var overall = Painted(child, root, root);
            if (focus(child) && overall.Width > 0 && overall.Height > 0 && !Contains(parentRect, painted))
            {
                problems.Add($"{Describe(child)} paints outside {Describe(visual)}: {Format(painted)} not within {Format(parentRect)}");
            }
        }

        if (visual is StackPanel or DockPanel or WrapPanel)
        {
            for (int i = 0; i < children.Count; i++)
            {
                for (int j = i + 1; j < children.Count; j++)
                {
                    var first = Painted(children[i], root, root);
                    var second = Painted(children[j], root, root);
                    var overlap = first.Intersect(second);
                    if (focus(children[i]) && focus(children[j]) && overlap.Width > Tolerance && overlap.Height > Tolerance)
                    {
                        problems.Add($"{Describe(children[i])} overlaps {Describe(children[j])} in {Describe(visual)}: {Format(overlap)}");
                    }
                }
            }
        }

        foreach (var child in children)
        {
            Visit(child, root, problems, focus);
        }
    }

    /// <summary>
    /// Describes the width of a text block that wraps its text into more than one line while being narrower than
    /// <see cref="MinimumWrapCharacters"/> characters of its font (average glyph about half the font size), or null.
    /// </summary>
    /// <param name="text">The text block, already measured and arranged.</param>
    public static string? FindExtremeWrap(TextBlock text)
    {
        if (text.TextWrapping == Avalonia.Media.TextWrapping.NoWrap || string.IsNullOrWhiteSpace(text.Text) || text.Text.Trim().Length <= MinimumWrapCharacters
            || text.TextLayout?.TextLines is not { Count: > 1 } lines)
        {
            return null;
        }

        var characters = text.Bounds.Width / Math.Max(1, text.FontSize * 0.5);
        return characters < MinimumWrapCharacters
            ? $"{text.Bounds.Width:0} px (about {characters:0.#} characters, {lines.Count} lines): \"{(text.Text.Length > 40 ? text.Text[..40] + "..." : text.Text)}\""
            : null;
    }

    private static bool Contains(Rect outer, Rect inner)
        => inner.Width <= 0 || inner.Height <= 0
            || (inner.Left >= outer.Left - Tolerance && inner.Top >= outer.Top - Tolerance && inner.Right <= outer.Right + Tolerance && inner.Bottom <= outer.Bottom + Tolerance);

    private static string Describe(Visual visual)
    {
        string name = visual is Control { Name: { Length: > 0 } controlName } ? "#" + controlName : "";
        var ancestors = new List<string>();
        int depth = 0;
        for (var ancestor = visual.GetVisualParent(); ancestor is not null && depth < 3; ancestor = ancestor.GetVisualParent(), depth++)
        {
            ancestors.Add(ancestor.GetType().Name);
        }

        return $"{visual.GetType().Name}{name} (in {string.Join('<', ancestors)})";
    }

    private static string Format(Rect rect) => $"[{rect.Left:0},{rect.Top:0} {rect.Width:0}x{rect.Height:0}]";
}
