using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Platform;

/// <summary>Applies the colour theme and accent palette.</summary>
internal interface IThemeService
{
    /// <summary>Switches theme and accent.</summary>
    public void Apply(AppTheme theme, string accent);
}
