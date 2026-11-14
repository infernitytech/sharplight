using System.Globalization;
using Avalonia.Data.Converters;

namespace Sharplight.Studio.Converters;

/// <summary>True when the value equals the parameter (compared as invariant text), e.g. to highlight the selected tab.</summary>
internal sealed class EqualsConverter : IValueConverter
{
    /// <summary>The shared instance.</summary>
    public static EqualsConverter Instance { get; } = new();

    /// <inheritdoc/>
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        => string.Equals(System.Convert.ToString(value, CultureInfo.InvariantCulture), System.Convert.ToString(parameter, CultureInfo.InvariantCulture), StringComparison.Ordinal);

    /// <inheritdoc/>
    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotSupportedException();
}
