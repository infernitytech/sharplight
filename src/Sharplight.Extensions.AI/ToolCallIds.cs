namespace Sharplight.Extensions.AI;

internal static class ToolCallIds
{
    public static string New() => "call_" + Guid.NewGuid().ToString("N")[..24];
}
