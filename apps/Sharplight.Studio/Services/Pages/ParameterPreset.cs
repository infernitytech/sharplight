namespace Sharplight.Studio.Services.Pages;

/// <summary>A named set of parameter values of an inference page.</summary>
/// <param name="Name">Shown in the preset picker.</param>
/// <param name="Values">The values by parameter key, as invariant text.</param>
/// <param name="BuiltIn">Shipped with the page (cannot be deleted).</param>
internal sealed record ParameterPreset(string Name, Dictionary<string, string> Values, bool BuiltIn = false);
