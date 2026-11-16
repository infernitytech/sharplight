namespace Sharplight.Models;

/// <summary>A companion file a model needs (<see cref="IModelImporter.DescribeCompanions"/>): its auxiliary role and how to present it.</summary>
/// <param name="Role">The auxiliary role it is loaded in (e.g. <see cref="Vision.AuxiliaryRoles.TextEncoder"/>).</param>
/// <param name="Label">What it is, for people (e.g. "Text encoder (CLIP-L)").</param>
/// <param name="Example">Typical file names (e.g. "clip_l.safetensors").</param>
public sealed record CompanionFile(string Role, string Label, string Example);
