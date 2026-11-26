using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.VisualTree;

namespace Sharplight.Studio.Views;

/// <summary>
/// Arrow keys in the navigation: Up and Down (Home and End) move the focus between the visible group headers and entries, Left collapses
/// a group (or moves from an entry to its group's header) and Right expands it. Enter and Space press the focused button as usual.
/// </summary>
internal static class NavigationKeyboard
{
    /// <summary>Handles <paramref name="e"/> when it is a navigation key pressed on a button of <paramref name="sidebar"/>.</summary>
    /// <param name="sidebar">The navigation's root.</param>
    /// <param name="e">The key press.</param>
    public static void Handle(Control sidebar, KeyEventArgs e)
    {
        if (e.KeyModifiers != KeyModifiers.None || e.Source is not Visual source)
        {
            return;
        }

        var focused = source as Button ?? source.FindAncestorOfType<Button>();
        if (focused is null)
        {
            return;
        }

        var buttons = Buttons(sidebar);
        int index = buttons.IndexOf(focused);
        if (index < 0)
        {
            return;
        }

        Button? target = e.Key switch
        {
            Key.Down => buttons[Math.Min(index + 1, buttons.Count - 1)],
            Key.Up => buttons[Math.Max(index - 1, 0)],
            Key.Home => buttons[0],
            Key.End => buttons[^1],
            Key.Left => Collapse(focused, buttons, index),
            Key.Right => Expand(focused),
            _ => null,
        };
        if (target is null)
        {
            return;
        }

        target.Focus(NavigationMethod.Directional);
        e.Handled = true;
    }

    /// <summary>The focusable navigation buttons in the order shown.</summary>
    private static List<Button> Buttons(Control sidebar)
        => [.. sidebar.GetVisualDescendants().OfType<Button>().Where(IsNavigation).Where(button => button.IsEffectivelyVisible && button.IsEffectivelyEnabled)];

    private static bool IsNavigation(Button button)
        => button.Classes.Contains("navgroup") || button.Classes.Contains("navitem") || button.Classes.Contains("navrail");

    /// <summary>Left on an expanded header collapses it; on an entry it moves to the entry's header.</summary>
    private static Button? Collapse(Button focused, List<Button> buttons, int index)
    {
        if (focused is ToggleButton { IsChecked: true } header)
        {
            header.IsChecked = false;
            return header;
        }

        if (!focused.Classes.Contains("navitem"))
        {
            return null;
        }

        for (int i = index - 1; i >= 0; i--)
        {
            if (buttons[i] is ToggleButton previous)
            {
                return previous;
            }
        }

        return null;
    }

    /// <summary>Right on a collapsed header expands it.</summary>
    private static Button? Expand(Button focused)
    {
        if (focused is ToggleButton { IsChecked: false } header)
        {
            header.IsChecked = true;
            return header;
        }

        return null;
    }
}
