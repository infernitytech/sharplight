namespace Sharplight.Studio.Services.Server;

/// <summary>One HTTP request the server answered.</summary>
/// <param name="Timestamp">When it finished.</param>
/// <param name="Method">HTTP method.</param>
/// <param name="Path">Request path.</param>
/// <param name="Status">Response status code.</param>
/// <param name="DurationMs">Time to the end of the response (streams included).</param>
/// <param name="Client">Remote address.</param>
internal sealed record HttpLogEntry(DateTimeOffset Timestamp, string Method, string Path, int Status, double DurationMs, string Client);
