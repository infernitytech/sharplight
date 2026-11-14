namespace Sharplight.Studio.Models;

/// <summary>One finished generation request, as stored in the statistics history.</summary>
internal sealed record RequestRecord
{
    /// <summary>Row id (0 until stored).</summary>
    public long Id { get; init; }

    /// <summary>When it finished.</summary>
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>The model id.</summary>
    public required string Model { get; init; }

    /// <summary>Chat or API.</summary>
    public RequestSource Source { get; init; }

    /// <summary>The device the model ran on.</summary>
    public string Device { get; init; } = "";

    /// <summary>Token counts and timings.</summary>
    public required MessageStats Stats { get; init; }
}
