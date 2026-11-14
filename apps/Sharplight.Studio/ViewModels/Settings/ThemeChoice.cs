using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Settings;

/// <summary>A theme in the picker.</summary>
/// <param name="Value">The theme.</param>
/// <param name="Name">Shown name.</param>
internal sealed record ThemeChoice(AppTheme Value, string Name)
{
    /// <summary>The choices.</summary>
    public static IReadOnlyList<ThemeChoice> All { get; } = [new(AppTheme.Dark, "Dark"), new(AppTheme.Light, "Light"), new(AppTheme.System, "System")];
}
