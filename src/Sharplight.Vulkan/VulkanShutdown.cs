namespace Sharplight.Vulkan;

/// <summary>
/// Releases the Vulkan devices still open when the process exits. A device that was never disposed keeps its queues, memory and
/// pipelines alive until the driver's own exit-time teardown, which can crash (SIGSEGV after <c>Main</c> returned) when it runs
/// against live devices; destroying them on <see cref="AppDomain.ProcessExit"/> first makes leaked devices harmless.
/// </summary>
internal static class VulkanShutdown
{
    private static readonly Lock s_lock = new();
    private static readonly HashSet<VulkanDevice> s_open = new(ReferenceEqualityComparer.Instance);
    private static bool s_hooked;

    /// <summary>Records an opened device until it is disposed.</summary>
    /// <param name="device">The device.</param>
    public static void Track(VulkanDevice device)
    {
        lock (s_lock)
        {
            if (!s_hooked)
            {
                AppDomain.CurrentDomain.ProcessExit += (_, _) => DisposeOpen();
                s_hooked = true;
            }

            s_open.Add(device);
        }
    }

    /// <summary>Forgets a disposed device.</summary>
    /// <param name="device">The device.</param>
    public static void Untrack(VulkanDevice device)
    {
        lock (s_lock)
        {
            s_open.Remove(device);
        }
    }

    private static void DisposeOpen()
    {
        VulkanDevice[] open;
        lock (s_lock)
        {
            open = [.. s_open];
        }

        foreach (var device in open)
        {
            try
            {
                device.Dispose();
            }
            catch (Exception)
            {
                // Best effort at exit: a device that cannot be torn down is left to the driver.
            }
        }
    }
}
