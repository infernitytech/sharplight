namespace Sharplight.Runtime;

/// <summary>One shared weight buffer of a <see cref="SharedWeights"/> registry: the provider's handle, holders and fill state.</summary>
internal sealed class SharedWeightEntry(object device, WeightBufferKey key, object buffer, long bytes, Action<object> free)
{
    public object Device { get; } = device;
    public WeightBufferKey Key { get; } = key;
    public object Buffer { get; } = buffer;
    public long Bytes { get; } = bytes;
    public Action<object> Free { get; } = free;
    /// <summary>Leases not yet disposed (guarded by the registry's lock).</summary>
    public int Holders { get; set; } = 1;
    /// <summary>Set once the filling holder wrote the data or gave up.</summary>
    public ManualResetEventSlim Ready { get; } = new(false);
    public bool Failed { get; private set; }
    /// <summary>Still findable by new acquirers (false after a failed fill or once freed).</summary>
    public bool Listed { get; set; } = true;

    public void MarkFilled(bool failed)
    {
        if (Ready.IsSet)
        {
            return;
        }

        Failed = failed;
        Ready.Set();
    }
}
