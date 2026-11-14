using System.Globalization;
using Avalonia;
using Avalonia.Data.Converters;
using Avalonia.Media;

namespace Sharplight.Studio.Converters;

/// <summary>Looks up an icon geometry by its resource key (e.g. a page's <c>Icon</c>).</summary>
internal sealed class IconConverter : IValueConverter
{
    /// <summary>The shared instance.</summary>
    public static IconConverter Instance { get; } = new();

    /// <inheritdoc/>
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return value is string key && Application.Current is { } app && app.TryGetResource(key, app.ActualThemeVariant, out var resource) && resource is Geometry geometry
            ? geometry
            : null;
    }

    /// <inheritdoc/>
    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotSupportedException();
}
