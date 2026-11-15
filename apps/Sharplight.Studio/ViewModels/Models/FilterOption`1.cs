namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A choice of a filter picker with the value it stands for.</summary>
/// <typeparam name="T">The value (null stands for "any" in the filters that have such a choice).</typeparam>
/// <param name="Name">Shown in the picker.</param>
/// <param name="Value">The value.</param>
internal sealed record FilterOption<T>(string Name, T Value) : FilterOption(Name);
