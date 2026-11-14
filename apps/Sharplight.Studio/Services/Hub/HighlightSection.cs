using Sharplight.Hub;

namespace Sharplight.Studio.Services.Hub;

/// <summary>One section of highlighted models, best first.</summary>
/// <param name="Kind">Which section.</param>
/// <param name="Models">The repositories, all with a supported architecture.</param>
internal sealed record HighlightSection(HighlightKind Kind, IReadOnlyList<HubModelSummary> Models);
