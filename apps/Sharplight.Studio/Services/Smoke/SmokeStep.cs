namespace Sharplight.Studio.Services.Smoke;

/// <summary>One step of the smoke run (<see cref="SmokeRunner"/>).</summary>
/// <param name="Name">The step's name: <c>devices</c>, <c>feed</c>, <c>install</c>, <c>decode</c>, <c>load</c>, <c>infer</c> or <c>unload</c>.</param>
/// <param name="Passed">Whether the step succeeded.</param>
/// <param name="Detail">What the step found, or why it failed.</param>
/// <param name="Seconds">How long the step took.</param>
internal sealed record SmokeStep(string Name, bool Passed, string Detail, double Seconds);
