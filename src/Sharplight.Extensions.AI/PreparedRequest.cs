using Sharplight.Llm;
using Sharplight.Vision;

namespace Sharplight.Extensions.AI;

/// <summary>The pieces of a request after mapping Microsoft.Extensions.AI types to Sharplight types.</summary>
internal sealed class PreparedRequest
{
    public required string Prompt { get; init; }
    public required IReadOnlyList<ImageInput> Images { get; init; }
    public required GenerationOptions Options { get; init; }
    public required ToolSchemas Tools { get; init; }
    public required bool HasTools { get; init; }
    /// <summary>The prompt ends inside an open reasoning block.</summary>
    public required bool StartsInReasoning { get; init; }
}
