namespace Sharplight.Models;

/// <summary>
/// A canonical architecture whose models are never loaded on their own but only as an auxiliary source of another model (e.g. a
/// speculative-decoding drafter as the <c>draft</c> source of its target); loading one on its own fails with a message that says so.
/// Registered with <c>services.AddAuxiliaryArchitecture(architecture, role)</c>.
/// </summary>
/// <param name="Architecture">The canonical architecture id.</param>
/// <param name="Role">The auxiliary role its models are loaded in (e.g. <see cref="Vision.AuxiliaryRoles.Draft"/>).</param>
public sealed record AuxiliaryArchitecture(string Architecture, string Role);
