using System.Diagnostics;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Services.Monitoring;

/// <summary>
/// <see cref="ISystemMonitor"/> from the process counters and the engines' metrics. GPU load is reported as the share of time the
/// schedulers spent in forward passes: the provider abstraction exposes device memory, not utilization counters.
/// </summary>
internal sealed class SystemMonitor : ISystemMonitor
{
    private readonly IModelRuntime _runtime;
    private readonly Process _process = Process.GetCurrentProcess();
    private readonly Dictionary<string, double> _stepSeconds = [];
    private long _lastTimestamp = Stopwatch.GetTimestamp();
    private TimeSpan _lastCpu;

    public SystemMonitor(IModelRuntime runtime)
    {
        _runtime = runtime;
        _lastCpu = _process.TotalProcessorTime;
    }

    /// <inheritdoc/>
    public SystemSample Sample()
    {
        _process.Refresh();
        long now = Stopwatch.GetTimestamp();
        double wall = Math.Max(1e-3, Stopwatch.GetElapsedTime(_lastTimestamp, now).TotalSeconds);
        var cpu = _process.TotalProcessorTime;
        double cpuPercent = (cpu - _lastCpu).TotalSeconds / wall / Environment.ProcessorCount * 100;
        _lastCpu = cpu;
        _lastTimestamp = now;

        double busy = 0;
        int active = 0;
        int queued = 0;
        double tokensPerSecond = 0;
        foreach (var model in _runtime.LoadedModels)
        {
            if (model.GetMetrics() is not { } metrics)
            {
                continue;
            }

            double seconds = metrics.StepDuration.Sum;
            double previous = _stepSeconds.GetValueOrDefault(model.Id, seconds);
            _stepSeconds[model.Id] = seconds;
            busy += Math.Max(0, seconds - previous) / wall;
            active += metrics.ActiveSequences;
            queued += metrics.QueueDepth;
            tokensPerSecond += metrics.RecentTokensPerSecond;
        }

        var memory = GC.GetGCMemoryInfo();
        return new SystemSample(DateTimeOffset.UtcNow, Math.Clamp(cpuPercent, 0, 100), _process.WorkingSet64, memory.TotalAvailableMemoryBytes,
            _runtime.GetDeviceUsage(), Math.Clamp(busy * 100, 0, 100), active, queued, tokensPerSecond);
    }
}
