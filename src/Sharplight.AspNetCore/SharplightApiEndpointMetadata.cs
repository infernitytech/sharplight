namespace Sharplight.AspNetCore;

/// <summary>Marks the endpoints that run through the Sharplight request pipeline (authentication and resolution are done there).</summary>
internal sealed class SharplightApiEndpointMetadata
{
    public static SharplightApiEndpointMetadata Instance { get; } = new();
}
