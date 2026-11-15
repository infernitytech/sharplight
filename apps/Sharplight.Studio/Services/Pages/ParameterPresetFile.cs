namespace Sharplight.Studio.Services.Pages;

/// <summary>The file a page keeps its parameter presets in (<c>presets.json</c>): the user's presets and the values of the last run.</summary>
internal sealed class ParameterPresetFile
{
    /// <summary>The presets the user saved.</summary>
    public List<ParameterPreset> Presets { get; set; } = [];

    /// <summary>The values the page had at its last run, restored when the page opens.</summary>
    public Dictionary<string, string> Last { get; set; } = [];
}
