using System.Globalization;
using Avalonia.Data.Converters;

namespace Sharplight.Studio.Converters;

/// <summary>The text of a two-step delete button: <c>Delete</c>, then <c>Click again to delete</c> once it was clicked.</summary>
internal sealed class ConfirmTextConverter : IValueConverter
{
    /// <summary>The shared instance.</summary>
    public static ConfirmTextConverter Instance { get; } = new();

    /// <inheritdoc/>
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture) => value is true ? "Click again to delete" : "Delete";

    /// <inheritdoc/>
    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotSupportedException();
}
