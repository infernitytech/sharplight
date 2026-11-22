using Sharplight.Hosting.Fit;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Settings;
using Sharplight.Studio.Services.Threading;

namespace Sharplight.Studio.Services.Fit;

/// <summary>
/// The devices' live memory figures (free, usable with Studio's own models unloaded, unified and system memory within the process's
/// cgroup limit, as the providers report them) for every fit estimate, kept fresh: read again on request (a picker or the load panel
/// opens), after models load or unload (<see cref="IModelRuntime.ModelsChanged"/>, at most once a second), and periodically while
/// a view that shows fits is visible (<see cref="Watch"/>). <see cref="Changed"/> tells those views when the figures moved.
/// </summary>
internal sealed class DeviceMemoryWatch : IDisposable
{
    /// <summary>The period of the refresh while a view watches.</summary>
    public static readonly TimeSpan DefaultInterval = TimeSpan.FromSeconds(5);

    /// <summary>A change smaller than this on every device is not reported (allocator noise, other applications' small buffers).</summary>
    public const long Tolerance = 64L << 20;

    private readonly IModelRuntime _runtime;
    private readonly ISettingsService _settings;
    private readonly TimeProvider _time;
    private readonly TimeSpan _interval;
    private readonly UpdateThrottle _modelsThrottle;
    private readonly EventHandler _modelsChanged;
    private readonly Lock _lock = new();
    private IReadOnlyList<DeviceMemory>? _snapshot;
    private Task<IReadOnlyList<DeviceMemory>>? _reading;
    private ITimer? _timer;
    private int _watchers;
    private bool _disposed;

    /// <param name="runtime">Reads the figures and reports loads and unloads.</param>
    /// <param name="settings">The disabled device features (for devices opened by a read).</param>
    /// <param name="time">The clock (tests); default the system clock.</param>
    /// <param name="interval">The period while watched; default <see cref="DefaultInterval"/>.</param>
    public DeviceMemoryWatch(IModelRuntime runtime, ISettingsService settings, TimeProvider? time = null, TimeSpan? interval = null)
    {
        _runtime = runtime;
        _settings = settings;
        _time = time ?? TimeProvider.System;
        _interval = interval ?? DefaultInterval;
        _modelsThrottle = new UpdateThrottle(() => _ = RefreshQuietlyAsync(), TimeSpan.FromSeconds(1), _time);
        _modelsChanged = (_, _) => _modelsThrottle.Request();
        runtime.ModelsChanged += _modelsChanged;
    }

    /// <summary>Raised (on any thread) with the new figures when a read found them changed by more than <see cref="Tolerance"/>.</summary>
    public event EventHandler<IReadOnlyList<DeviceMemory>>? Changed;

    /// <summary>The latest figures, or null before the first read.</summary>
    public IReadOnlyList<DeviceMemory>? Snapshot
    {
        get
        {
            lock (_lock)
            {
                return _snapshot;
            }
        }
    }

    /// <summary>When <see cref="Snapshot"/> was read.</summary>
    public DateTimeOffset? TakenAt { get; private set; }

    /// <summary>How many views watch (for tests).</summary>
    internal int Watchers => Volatile.Read(ref _watchers);

    /// <summary>Reads the figures now (a read in progress is shared) and raises <see cref="Changed"/> when they moved.</summary>
    /// <param name="cancellationToken">Stops waiting for the read (the read itself completes and is kept).</param>
    public Task<IReadOnlyList<DeviceMemory>> RefreshAsync(CancellationToken cancellationToken = default)
    {
        Task<IReadOnlyList<DeviceMemory>> reading;
        lock (_lock)
        {
            // On the thread pool: the read clears _reading under the lock when it ends, after this assignment.
            reading = _reading ??= Task.Run(ReadAsync, CancellationToken.None);
        }

        return reading.WaitAsync(cancellationToken);
    }

    /// <summary>
    /// Keeps the figures fresh while the returned handle is not disposed: a read now and every <see cref="DefaultInterval"/>. Views that
    /// show fit estimates hold one while visible.
    /// </summary>
    public IDisposable Watch()
    {
        lock (_lock)
        {
            if (!_disposed && _watchers++ == 0)
            {
                // One shot, armed again on every tick: no ticks pile up behind a slow read.
                _timer = _time.CreateTimer(_ => Tick(), null, _interval, Timeout.InfiniteTimeSpan);
            }
        }

        _ = RefreshQuietlyAsync();
        return new Handle(this);
    }

    /// <summary>Stops following the runtime and the timer.</summary>
    public void Dispose()
    {
        lock (_lock)
        {
            _disposed = true;
            _timer?.Dispose();
            _timer = null;
        }

        _runtime.ModelsChanged -= _modelsChanged;
        _modelsThrottle.Dispose();
    }

    /// <summary>True when every device of <paramref name="next"/> is within <see cref="Tolerance"/> of <paramref name="previous"/>.</summary>
    /// <param name="previous">The earlier figures.</param>
    /// <param name="next">The new figures.</param>
    internal static bool Same(IReadOnlyList<DeviceMemory>? previous, IReadOnlyList<DeviceMemory> next)
    {
        if (previous is null || previous.Count != next.Count)
        {
            return false;
        }

        for (int i = 0; i < next.Count; i++)
        {
            var a = previous[i];
            var b = next[i];
            if (a.Id != b.Id || Math.Abs(a.FreeBytes - b.FreeBytes) > Tolerance || Math.Abs(a.UsableBytes - b.UsableBytes) > Tolerance
                || Math.Abs(a.TotalBytes - b.TotalBytes) > Tolerance)
            {
                return false;
            }
        }

        return true;
    }

    private void Tick()
    {
        lock (_lock)
        {
            _timer?.Change(_interval, Timeout.InfiniteTimeSpan);
        }

        _ = RefreshQuietlyAsync();
    }

    private void Unwatch()
    {
        lock (_lock)
        {
            if (_watchers > 0 && --_watchers == 0)
            {
                _timer?.Dispose();
                _timer = null;
            }
        }
    }

    private async Task<IReadOnlyList<DeviceMemory>> ReadAsync()
    {
        try
        {
            var memory = await _runtime.GetDeviceMemoryAsync(_settings.Current.DisabledFeatures, CancellationToken.None).ConfigureAwait(false);
            bool changed;
            lock (_lock)
            {
                changed = !Same(_snapshot, memory);
                _snapshot = memory;
                TakenAt = _time.GetUtcNow();
            }

            if (changed)
            {
                Changed?.Invoke(this, memory);
            }

            return memory;
        }
        finally
        {
            lock (_lock)
            {
                _reading = null;
            }
        }
    }

    private async Task RefreshQuietlyAsync()
    {
        if (_disposed)
        {
            return;
        }

        try
        {
            await RefreshAsync().ConfigureAwait(false);
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            // A device that cannot be asked now keeps its last figures; the next read tries again.
        }
    }

    private sealed class Handle(DeviceMemoryWatch owner) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                owner.Unwatch();
            }
        }
    }
}
