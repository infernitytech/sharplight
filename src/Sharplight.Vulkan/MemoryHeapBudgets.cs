namespace Sharplight.Vulkan;

/// <summary>Per-heap figures of VK_EXT_memory_budget: what this process may use of each heap (accounting for other processes) and what it uses.</summary>
/// <param name="Budget">Bytes per heap index; empty when the extension is not enabled.</param>
/// <param name="Usage">Bytes per heap index this process uses.</param>
internal sealed record MemoryHeapBudgets(ulong[] Budget, ulong[] Usage)
{
    /// <summary>True when the figures come from the driver.</summary>
    public bool IsKnown => Budget.Length > 0;
}
