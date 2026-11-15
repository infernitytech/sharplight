using Sharplight.Hub;

namespace Sharplight.Studio.Models;

/// <summary>How Studio presents one model kind (the <c>kind</c> of a model description): its name, icon and whether the engine runs it.</summary>
/// <param name="Id">The kind as a model description names it (<c>text-generation</c>, <c>image-generation</c>, ...).</param>
/// <param name="Title">The name shown in badges and filters.</param>
/// <param name="Icon">Icon resource key (a geometry in <c>Icons.axaml</c>).</param>
/// <param name="Tasks">The Hub tasks whose models have this kind.</param>
/// <param name="Runnable">The engine has a task client for this kind, so Studio can load and run these models.</param>
internal sealed record ModelKindInfo(string Id, string Title, string Icon, IReadOnlyList<ModelTask> Tasks, bool Runnable);
