using System.Globalization;
using Avalonia.Data.Converters;

namespace Sharplight.Studio.Converters;

/// <summary>True for a number greater than zero (e.g. to show a count badge).</summary>
internal sealed class PositiveConverter : IValueConverter
{
    /// <summary>The shared instance.</summary>
    public static PositiveConverter Instance { get; } = new();

    /// <inheritdoc/>
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture) => value switch
    {
        int number => number > 0,
        long number => number > 0,
        double number => number > 0,
        _ => false,
    };

    /// <inheritdoc/>
    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotSupportedException();
}
