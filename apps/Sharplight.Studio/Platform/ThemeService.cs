using Avalonia;
using Avalonia.Media;
using Avalonia.Styling;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.Platform;

/// <summary>Switches the theme variant and replaces the accent colours (Fluent's <c>SystemAccentColor*</c> and the app's accent brushes).</summary>
internal sealed class ThemeService(Application application) : IThemeService
{
    /// <inheritdoc/>
    public void Apply(AppTheme theme, string accent)
    {
        application.RequestedThemeVariant = theme switch
        {
            AppTheme.Light => ThemeVariant.Light,
            AppTheme.Dark => ThemeVariant.Dark,
            _ => ThemeVariant.Default,
        };

        var palette = AccentPalette.Get(accent);
        var resources = application.Resources;
        resources["SystemAccentColor"] = Color.Parse(palette.Accent);
        resources["SystemAccentColorLight1"] = Color.Parse(palette.Light1);
        resources["SystemAccentColorLight2"] = Color.Parse(palette.Light2);
        resources["SystemAccentColorLight3"] = Color.Parse(palette.Light3);
        resources["SystemAccentColorDark1"] = Color.Parse(palette.Dark1);
        resources["SystemAccentColorDark2"] = Color.Parse(palette.Dark2);
        resources["SystemAccentColorDark3"] = Color.Parse(palette.Dark3);
        resources["AccentBrush"] = new SolidColorBrush(Color.Parse(palette.Accent));
        resources["AccentSecondaryBrush"] = new SolidColorBrush(Color.Parse(palette.Secondary));
        resources["AccentSoftBrush"] = new SolidColorBrush(Color.Parse(palette.Accent), 0.16);
        resources["AccentGradientBrush"] = new LinearGradientBrush
        {
            StartPoint = new RelativePoint(0, 0, RelativeUnit.Relative),
            EndPoint = new RelativePoint(1, 1, RelativeUnit.Relative),
            GradientStops = [new GradientStop(Color.Parse(palette.Accent), 0), new GradientStop(Color.Parse(palette.Secondary), 1)],
        };
    }
}
