namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A choice of a filter picker: the name the views show (<see cref="FilterOption{T}"/> adds the value it stands for).</summary>
/// <param name="Name">Shown in the picker.</param>
internal abstract record FilterOption(string Name);
