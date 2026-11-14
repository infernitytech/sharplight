using Sharplight.Models;

namespace Sharplight.Runtime;

/// <summary>
/// Device-resident weight buffers shared by the compiled models of one model source (for example a decoder and the draft head
/// that reads the same embedding and LM head, or several sessions of one model): pass the same instance as
/// <see cref="CompileOptions.SharedWeights"/> to every compilation, and providers that support it
/// (<see cref="IComputeDevice.SharesWeights"/>) upload each tensor once per device and layout instead of once per compiled model.
/// Buffers are keyed by device and <see cref="WeightBufferKey"/> and reference counted: the last compiled segment holding a
/// buffer frees it. Thread-safe.
/// </summary>
/// <remarks>
/// Provider contract: when a compilation finds a non-null <see cref="CompileOptions.SharedWeights"/>, the provider takes each
/// weight buffer through <see cref="Acquire{T}"/> (key from <see cref="WeightBufferKey.For"/> plus its device layout), fills
/// it when the lease says <see cref="SharedWeightLease{T}.MustFill"/> and marks it <see cref="SharedWeightLease{T}.Filled"/>,
/// waits (<see cref="SharedWeightLease{T}.WaitFilled"/>) for buffers another compilation fills, and disposes its leases instead
/// of freeing the buffers. Weights must be read-only on the device. A segment's <see cref="ICompiledSegment.DeviceMemoryBytes"/>
/// counts the shared buffers it allocated, not the ones it found.
/// </remarks>
public sealed class SharedWeights
{
    private readonly Dictionary<(object Device, WeightBufferKey Key), SharedWeightEntry> _entries = [];
    private readonly Dictionary<IWeightSource, int> _residentSources = new(ReferenceEqualityComparer.Instance);
    private readonly Lock _lock = new();
    private long _bytes;

    /// <summary>Bytes of all buffers currently held (each counted once, whatever the number of holders).</summary>
    public long ResidentBytes
    {
        get
        {
            lock (_lock)
            {
                return _bytes;
            }
        }
    }

    /// <summary>Number of buffers currently held.</summary>
    public int Count
    {
        get
        {
            lock (_lock)
            {
                return _entries.Count;
            }
        }
    }

    /// <summary>
    /// Takes a hold on the buffer of <paramref name="key"/> on <paramref name="device"/>, allocating it with
    /// <paramref name="allocate"/> when no compilation holds it (the returned lease then has to fill it).
    /// </summary>
    /// <typeparam name="T">The provider's buffer handle.</typeparam>
    /// <param name="device">The device the buffer lives on (an <see cref="IComputeDevice"/> or a provider object identifying it).</param>
    /// <param name="key">The tensor and its device layout.</param>
    /// <param name="bytes">Device bytes of the buffer (for <see cref="ResidentBytes"/>).</param>
    /// <param name="allocate">Allocates the buffer (called under the registry's lock: allocation only, no upload).</param>
    /// <param name="free">Frees the buffer once the last lease is disposed.</param>
    public SharedWeightLease<T> Acquire<T>(object device, WeightBufferKey key, long bytes, Func<T> allocate, Action<T> free) where T : class
    {
        ArgumentNullException.ThrowIfNull(device);
        lock (_lock)
        {
            if (_entries.TryGetValue((device, key), out var entry))
            {
                if (entry.Buffer is not T)
                {
                    throw new InvalidOperationException($"Shared weight buffer of {key.Source} has handle type {entry.Buffer.GetType().Name}, not {typeof(T).Name}.");
                }

                entry.Holders++;
                return new SharedWeightLease<T>(this, entry, mustFill: false);
            }

            var buffer = allocate();
            entry = new SharedWeightEntry(device, key, buffer, bytes, b => free((T)b));
            _entries[(device, key)] = entry;
            _residentSources[key.Source] = _residentSources.GetValueOrDefault(key.Source) + 1;
            _bytes += bytes;
            return new SharedWeightLease<T>(this, entry, mustFill: true);
        }
    }

    /// <summary>True when some device holds a buffer of <paramref name="weight"/>'s bytes (in any layout).</summary>
    public bool IsResident(WeightInfo weight)
    {
        lock (_lock)
        {
            return _residentSources.ContainsKey(weight.Source);
        }
    }

    /// <summary>A lease was disposed: frees the buffer with its last holder; a filler that never filled marks the buffer failed.</summary>
    internal void Release(SharedWeightEntry entry, bool filler)
    {
        bool free;
        lock (_lock)
        {
            if (filler && !entry.Ready.IsSet)
            {
                // Holders that already found the buffer see the failure; new acquirers allocate a fresh one.
                entry.MarkFilled(failed: true);
                Unlist(entry);
            }

            entry.Holders--;
            free = entry.Holders == 0;
            if (free)
            {
                Unlist(entry);
            }
        }

        if (free)
        {
            entry.Free(entry.Buffer);
            entry.Ready.Dispose();
        }
    }

    private void Unlist(SharedWeightEntry entry)
    {
        if (!entry.Listed)
        {
            return;
        }

        entry.Listed = false;
        _entries.Remove((entry.Device, entry.Key));
        _bytes -= entry.Bytes;
        int n = _residentSources[entry.Key.Source] - 1;
        if (n == 0)
        {
            _residentSources.Remove(entry.Key.Source);
        }
        else
        {
            _residentSources[entry.Key.Source] = n;
        }
    }
}
