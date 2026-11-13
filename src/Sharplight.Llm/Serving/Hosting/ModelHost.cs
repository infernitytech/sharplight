using System.Text;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>
/// Registry and memory-aware host for several models sharing the same devices. Models are registered by name and loaded eagerly
/// or on first use. Before a model is loaded its footprint is estimated (<see cref="IPreparedModel.Requirements"/>) and checked
/// against the live free memory of each pool minus <see cref="ModelHostOptions.ReserveBytes"/>; if it does not fit, the least
/// recently used idle models are unloaded, then (if the loader can) layers are offloaded to the fallback device, else the
/// request fails with <see cref="ModelDoesNotFitException"/>. Models idle for <see cref="ModelHostOptions.IdleTimeout"/> are unloaded.
/// Loads are serialized (so accounting never races); inference on loaded models is fully concurrent.
/// </summary>
public sealed class ModelHost : IDisposable
{
    private sealed class Entry(ModelRegistration reg)
    {
        public readonly ModelRegistration Reg = reg;
        public ModelState State;
        public LoadedModel? Instance;
        public IReadOnlyList<MemoryRequirement> Requirements = [];
        public long Estimated;
        public int Leases;
        public bool Draining;
        public DateTimeOffset? LoadedAt, LastUsed;
        public Task? LoadTask;
        public string? Error;
        public TaskCompletionSource Changed = new(TaskCreationOptions.RunContinuationsAsynchronously);
    }

    private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);
    private readonly List<string> _order = [];
    private readonly object _gate = new();
    private readonly SemaphoreSlim _loadGate = new(1, 1);
    private readonly ModelHostOptions _opt;
    private ITimer? _timer;
    private bool _disposed;

    public ModelHost(ModelHostOptions? options = null)
    {
        _opt = options ?? new();
        if (_opt.IdleTimeout is not null)
        {
            EnsureTimer();
        }
    }

    private void EnsureTimer()
    {
        lock (_gate)
        {
            _timer ??= _opt.Time.CreateTimer(_ => { try { SweepIdle(); } catch (Exception e) { _opt.Log?.Invoke("idle sweep failed: " + e.Message); } }, null, _opt.IdleCheckInterval, _opt.IdleCheckInterval);
        }
    }

    public ModelHostOptions Options => _opt;

    // ---- registry ---------------------------------------------------------------------------------------------------

    public void Register(ModelRegistration registration)
    {
        if (string.IsNullOrWhiteSpace(registration.Name))
        {
            throw new ArgumentException("A model needs a name.");
        }

        lock (_gate)
        {
            if (!_entries.TryAdd(registration.Name, new Entry(registration)))
            {
                throw new InvalidOperationException($"A model named '{registration.Name}' is already registered.");
            }

            _order.Add(registration.Name);
        }
        if (registration.IdleTimeout is not null)
        {
            EnsureTimer();
        }
    }

    /// <summary>Registers a model that is already loaded (kept loaded; <paramref name="instance"/> is disposed with the host unless <paramref name="pinned"/> models are left to the caller).</summary>
    public void Adopt(string name, LoadedModel instance, string? description = null)
    {
        Register(new ModelRegistration { Name = name, Loader = new AdoptedLoader(instance.Kind), Pinned = true, Description = description });
        lock (_gate)
        {
            var e = _entries[name];
            e.Instance = instance;
            e.LoadedAt = e.LastUsed = _opt.Time.GetUtcNow();
            SetState(e, ModelState.Loaded);
        }
    }

    private sealed class AdoptedLoader(ModelKind kind) : IModelLoader
    {
        public ModelKind Kind => kind;
        public IPreparedModel Prepare(CancellationToken ct) => throw new InvalidOperationException("An adopted model cannot be reloaded after it was unloaded.");
    }

    public bool Contains(string name) { lock (_gate)
        {
            return _entries.ContainsKey(name);
        }
    }

    /// <summary>Name of the first registered model of <paramref name="kind"/> (used when a request names no model).</summary>
    public string? DefaultName(ModelKind kind = ModelKind.Chat)
    {
        lock (_gate)
        {
            return _order.FirstOrDefault(n => _entries[n].Reg.Kind == kind);
        }
    }

    public IReadOnlyList<ModelInfo> List() { lock (_gate)
        {
            return [.. _order.Select(n => Snapshot(_entries[n]))];
        }
    }

    /// <summary>Currently loaded models and their instances (for metrics; instances may be unloaded at any time, hold a lease to use one).</summary>
    public IReadOnlyList<(string Name, LoadedModel Instance)> LoadedInstances()
    {
        lock (_gate)
        {
            return [.. _order.Select(n => _entries[n]).Where(e => e.State == ModelState.Loaded && e.Instance is not null).Select(e => (e.Reg.Name, e.Instance!))];
        }
    }

    public ModelInfo? Get(string name) { lock (_gate)
        {
            return _entries.TryGetValue(name, out var e) ? Snapshot(e) : null;
        }
    }

    private ModelInfo Snapshot(Entry e) => new(e.Reg.Name, e.Reg.Kind, e.State, e.Reg.Pinned, e.Reg.LoadAtStartup,
        e.Instance?.ResidentBytes is > 0 and var r ? r : 0, e.Estimated, e.Leases, e.LoadedAt, e.LastUsed, e.Error, e.Reg.Description);

    /// <summary>Current memory figures of every pool any registered model has used or declared (after at least one load attempt).</summary>
    public IReadOnlyList<(IMemoryDomain Domain, Sharplight.Runtime.DeviceMemoryInfo Info)> MemoryPools()
    {
        lock (_gate)
        {
            return [.. _entries.Values.SelectMany(e => e.Requirements).Select(r => r.Domain).DistinctBy(d => d.Id).Select(d => (d, d.GetMemoryInfo()))];
        }
    }

    // ---- acquiring ----------------------------------------------------------------------------------------------------

    /// <summary>
    /// Returns a lease on the model, loading it first when needed (waiting for a load already in progress). Throws
    /// <see cref="ModelNotFoundException"/> for an unknown name and <see cref="ModelDoesNotFitException"/> when it cannot be made to fit.
    /// </summary>
    public async Task<ModelLease> AcquireAsync(string name, ModelKind? kind = null, CancellationToken ct = default)
    {
        Entry e;
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_entries.TryGetValue(name, out e!))
            {
                throw new ModelNotFoundException(name);
            }
        }
        if (kind is { } k && e.Reg.Kind != k)
        {
            throw new ModelKindMismatchException(name, e.Reg.Kind, k);
        }

        while (true)
        {
            ct.ThrowIfCancellationRequested();
            Task wait;
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (e.State == ModelState.Loaded && !e.Draining)
                {
                    e.Leases++;
                    e.LastUsed = _opt.Time.GetUtcNow();
                    return new ModelLease(name, e.Instance!, () => Release(e));
                }
                if (e.State == ModelState.Loading)
                {
                    wait = e.LoadTask!;
                }
                else if (e.State is ModelState.Unloaded or ModelState.Failed)
                {
                    SetState(e, ModelState.Loading);
                    e.Error = null;
                    wait = e.LoadTask = Task.Run(() => DoLoad(e), CancellationToken.None);
                }
                else
                {
                    wait = e.Changed.Task;   // loaded but draining, or unloading: wait, then retry (which reloads)
                }
            }
            await wait.WaitAsync(ct).ConfigureAwait(false);
        }
    }

    private void Release(Entry e)
    {
        lock (_gate)
        {
            e.Leases--;
            e.LastUsed = _opt.Time.GetUtcNow();
            Signal(e);
        }
    }

    // caller holds _gate
    private void SetState(Entry e, ModelState s) { e.State = s; Signal(e); }
    private void Signal(Entry e) { var old = e.Changed; e.Changed = new(TaskCreationOptions.RunContinuationsAsynchronously); old.TrySetResult(); }

    // ---- loading ----------------------------------------------------------------------------------------------------

    /// <summary>Loads a model now (no-op when loaded).</summary>
    public async Task LoadAsync(string name, CancellationToken ct = default)
    {
        using var lease = await AcquireAsync(name, null, ct).ConfigureAwait(false);
    }

    /// <summary>Loads every model registered with <see cref="ModelRegistration.LoadAtStartup"/>, one after another; throws on the first that cannot load.</summary>
    public async Task LoadStartupModelsAsync(CancellationToken ct = default)
    {
        string[] names;
        lock (_gate)
        {
            names = [.. _order.Where(n => _entries[n].Reg.LoadAtStartup)];
        }

        foreach (var n in names)
        {
            await LoadAsync(n, ct).ConfigureAwait(false);
        }
    }

    private void DoLoad(Entry e)
    {
        _loadGate.Wait();
        IPreparedModel? prepared = null;
        try
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            prepared = e.Reg.Loader.Prepare(CancellationToken.None);
            long? limit = Admit(e, prepared);
            var instance = prepared.Load(limit);
            lock (_gate)
            {
                e.Instance = instance;
                e.Requirements = prepared.Requirements;
                e.Estimated = prepared.Requirements.Sum(r => r.Bytes);
                e.LoadedAt = e.LastUsed = _opt.Time.GetUtcNow();
                SetState(e, ModelState.Loaded);
            }
            _opt.Log?.Invoke($"Model '{e.Reg.Name}' loaded in {sw.ElapsedMilliseconds} ms (estimated {e.Estimated / 1048576.0:F0} MiB, resident {instance.ResidentBytes / 1048576.0:F0} MiB{(limit is null ? "" : ", offloaded to fit")}).");
        }
        catch (Exception ex)
        {
            lock (_gate)
            {
                e.Error = ex.Message;
                if (prepared is not null) { e.Requirements = prepared.Requirements; e.Estimated = prepared.Requirements.Sum(r => r.Bytes); }
                SetState(e, ModelState.Failed);
            }
            _opt.Log?.Invoke($"Model '{e.Reg.Name}' failed to load: {ex.Message}");
            throw;
        }
        finally
        {
            prepared?.Dispose();
            _loadGate.Release();
        }
    }

    /// <summary>Makes room for <paramref name="p"/> (evicting idle models); returns the primary-device limit when the model must be loaded partly offloaded.</summary>
    private long? Admit(Entry self, IPreparedModel p)
    {
        string name = self.Reg.Name;
        if (_opt.MaxLoadedModels > 0)
        {
            while (LoadedCount() >= _opt.MaxLoadedModels)
            {
                if (!EvictOne(null, self))
                {
                    throw new ModelBusyException(name, $"{LoadedCount()} models are loaded (limit {_opt.MaxLoadedModels}) and none of them is idle: {Busy()}");
                }
            }
        }

        long? limit = null;
        for (int i = 0; i < p.Requirements.Count; i++)
        {
            var req = p.Requirements[i];
            while (true)
            {
                long avail = req.Domain.GetMemoryInfo().Available - _opt.ReserveBytes;
                if (avail >= req.Bytes)
                {
                    break;
                }

                if (EvictOne(req.Domain, self))
                {
                    continue;
                }

                if (i == 0 && p.CanOffload)
                {
                    limit = Math.Max(0, avail);
                    _opt.Log?.Invoke($"Model '{name}' needs {req.Bytes / 1048576.0:F0} MiB on {req.Domain.Name} but only {Math.Max(0, avail) / 1048576.0:F0} MiB are free; offloading layers.");
                    break;
                }
                throw new ModelDoesNotFitException(name, req.Bytes, avail, _opt.ReserveBytes,
                    $"Pool {req.Domain.Name}: {req.Domain.GetMemoryInfo()}. Loaded: {Loaded()}. Free memory (stop other models or applications), lower the context/slots, or configure a CPU fallback to offload layers.");
            }
        }
        return limit;
    }

    private int LoadedCount() { lock (_gate)
        {
            return _entries.Values.Count(x => x.State is ModelState.Loaded or ModelState.Unloading);
        }
    }
    private string Busy() { lock (_gate)
        {
            return string.Join(", ", _entries.Values.Where(x => x.State == ModelState.Loaded).Select(x => $"{x.Reg.Name} ({x.Leases} active)"));
        }
    }
    private string Loaded()
    {
        lock (_gate)
        {
            var l = _entries.Values.Where(x => x.State == ModelState.Loaded).Select(x => $"{x.Reg.Name} {x.Instance!.ResidentBytes / 1048576} MiB{(x.Leases > 0 || x.Reg.Pinned ? " [in use]" : "")}").ToList();
            return l.Count == 0 ? "none" : string.Join(", ", l);
        }
    }

    /// <summary>Unloads the least recently used evictable model (optionally one holding memory in <paramref name="domain"/>). False when there is none.</summary>
    private bool EvictOne(IMemoryDomain? domain, Entry requester)
    {
        Entry? victim = null;
        lock (_gate)
        {
            foreach (var x in _entries.Values)
            {
                if (x == requester || x.State != ModelState.Loaded || x.Leases > 0 || x.Draining || x.Reg.Pinned || !x.Instance!.IsIdle)
                {
                    continue;
                }

                if (domain is not null && !x.Requirements.Any(r => r.Domain.Id == domain.Id))
                {
                    continue;
                }

                if (victim is null || x.LastUsed < victim.LastUsed)
                {
                    victim = x;
                }
            }
            if (victim is null)
            {
                return false;
            }

            SetState(victim, ModelState.Unloading);
        }
        _opt.Log?.Invoke($"Evicting model '{victim.Reg.Name}' to free memory.");
        FinishUnload(victim);
        return true;
    }

    private void FinishUnload(Entry e)
    {
        LoadedModel? inst;
        lock (_gate)
        {
            inst = e.Instance;
        }

        try { inst?.Dispose(); }
        finally
        {
            lock (_gate)
            {
                e.Instance = null;
                e.Draining = false;
                SetState(e, ModelState.Unloaded);
            }
        }
    }

    // ---- unloading --------------------------------------------------------------------------------------------------

    /// <summary>
    /// Unloads a model and frees its device memory. New requests wait while running ones drain (up to <paramref name="drainTimeout"/>,
    /// default <see cref="ModelHostOptions.UnloadDrainTimeout"/>); if they do not finish, <paramref name="force"/> cuts them off,
    /// otherwise <see cref="ModelBusyException"/> is thrown and the model stays loaded. Returns false when it was not loaded.
    /// </summary>
    public async Task<bool> UnloadAsync(string name, TimeSpan? drainTimeout = null, bool force = false)
    {
        Entry e;
        lock (_gate) { if (!_entries.TryGetValue(name, out e!))
            {
                throw new ModelNotFoundException(name);
            }
        }
        // Wait out a load in progress.
        Task? loading;
        lock (_gate)
        {
            loading = e.State == ModelState.Loading ? e.LoadTask : null;
        }

        if (loading is not null) { try { await loading.ConfigureAwait(false); } catch { } }
        var deadline = _opt.Time.GetUtcNow() + (drainTimeout ?? _opt.UnloadDrainTimeout);
        lock (_gate)
        {
            if (e.State != ModelState.Loaded)
            {
                return false;
            }

            if (e.Reg.Pinned && !force)
            {
                throw new ModelBusyException(name, "it is pinned.");
            }

            e.Draining = true;
            Signal(e);
        }
        try
        {
            while (true)
            {
                Task wait;
                lock (_gate) { if (e.Leases == 0) { SetState(e, ModelState.Unloading); break; } wait = e.Changed.Task; }
                var left = deadline - _opt.Time.GetUtcNow();
                if (left <= TimeSpan.Zero || await Task.WhenAny(wait, Task.Delay(left)).ConfigureAwait(false) != wait)
                {
                    lock (_gate) { if (e.Leases == 0) { SetState(e, ModelState.Unloading); break; } }
                    if (!force)
                    {
                        throw new ModelBusyException(name, $"{e.Leases} request(s) still running.");
                    }

                    lock (_gate)
                    {
                        SetState(e, ModelState.Unloading);
                    }

                    break;
                }
            }
        }
        catch
        {
            lock (_gate) { if (e.State == ModelState.Loaded) { e.Draining = false; Signal(e); } }
            throw;
        }
        await Task.Run(() => FinishUnload(e)).ConfigureAwait(false);
        return true;
    }

    /// <summary>Unloads every model that has been idle longer than its timeout. Called by the host's timer; public so it can be driven explicitly.</summary>
    public int SweepIdle()
    {
        var now = _opt.Time.GetUtcNow();
        List<Entry> victims = [];
        lock (_gate)
        {
            foreach (var x in _entries.Values)
            {
                var to = x.Reg.IdleTimeout ?? _opt.IdleTimeout;
                if (to is null || x.State != ModelState.Loaded || x.Leases > 0 || x.Draining || x.Reg.Pinned || !x.Instance!.IsIdle)
                {
                    continue;
                }

                if (now - (x.LastUsed ?? x.LoadedAt ?? now) < to)
                {
                    continue;
                }

                SetState(x, ModelState.Unloading);
                victims.Add(x);
            }
        }
        foreach (var v in victims) { _opt.Log?.Invoke($"Unloading idle model '{v.Reg.Name}'."); FinishUnload(v); }
        return victims.Count;
    }

    // ---- reporting / lifetime -----------------------------------------------------------------------------------------

    /// <summary>One-paragraph description of the models and memory pools (for logs and admin output).</summary>
    public string Describe()
    {
        var sb = new StringBuilder();
        foreach (var m in List())
        {
            sb.AppendLine($"  {m.Name} [{m.Kind}] {m.State}{(m.State == ModelState.Loaded ? $" {m.ResidentBytes / 1048576} MiB" : "")}{(m.Error is null ? "" : " error: " + m.Error)}");
        }

        foreach (var (d, i) in MemoryPools())
        {
            sb.AppendLine($"  {d.Name}: {i}");
        }

        return sb.ToString().TrimEnd();
    }

    /// <summary>Drains every loaded model (waiting up to <paramref name="timeout"/> for running requests), so a host can stop gracefully.</summary>
    public async Task DrainAsync(TimeSpan timeout)
    {
        List<LoadedModel> all;
        lock (_gate)
        {
            all = [.. _entries.Values.Where(x => x.Instance is not null).Select(x => x.Instance!)];
        }

        await Task.WhenAll(all.Select(m => m.DrainAsync(timeout))).ConfigureAwait(false);
    }

    public void Dispose()
    {
        List<Entry> all;
        lock (_gate) { if (_disposed) { return; } _disposed = true; all = [.. _entries.Values]; }
        _timer?.Dispose();
        _loadGate.Wait();   // let a load in progress finish
        try
        {
            foreach (var e in all)
            {
                LoadedModel? inst;
                lock (_gate) { inst = e.Instance; e.Instance = null; e.State = ModelState.Unloaded; Signal(e); }
                inst?.Dispose();
            }
        }
        finally { _loadGate.Release(); }
    }
}
