using Avalonia;
using Avalonia.Input;

namespace Sharplight.Studio.Controls;

/// <summary>A pointer event on a picture, with the position as fractions of the picture (0 to 1).</summary>
/// <param name="Position">The position as fractions of the picture's width and height.</param>
/// <param name="IsSecondary">The secondary button was pressed.</param>
/// <param name="Modifiers">The keyboard modifiers.</param>
internal sealed record ImagePointerEventArgs(Point Position, bool IsSecondary, KeyModifiers Modifiers);
