namespace Sharplight.AspNetCore;

/// <summary>
/// Receives the usage of every API request (tokens in and out, latency, scope, principal) after its response is complete, e.g. to
/// write usage records or enforce quotas. Register any number in the container, or one per route group
/// (<see cref="SharplightEndpointOptions.UsageObserver"/>). Called on the request's thread; exceptions are caught and ignored.
/// </summary>
public interface ISharplightUsageObserver
{
    /// <summary>Called once per request.</summary>
    /// <param name="usage">What the request used.</param>
    public void OnRequestCompleted(SharplightUsage usage);
}
