namespace Sharplight.AgentHost;

/// <summary>Command-line choices of the agent host.</summary>
/// <param name="ModelPath">Model file or directory.</param>
/// <param name="UseCpu">Run on the CPU provider.</param>
/// <param name="DeviceIndex">Device index within the provider.</param>
/// <param name="ContextLength">Context length in tokens.</param>
/// <param name="Urls">Address to listen on, or null for the ASP.NET Core default.</param>
internal sealed record AgentHostSettings(string ModelPath, bool UseCpu, int DeviceIndex, int ContextLength, string? Urls);
