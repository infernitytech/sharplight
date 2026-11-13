using System.Collections.Concurrent;
using SlangShaderSharp;

namespace Sharplight.Shaders;

/// <summary>
/// Process-wide pool of background threads that run Slang compilations. A Slang global session is not thread-safe, so every
/// worker owns a private one (created on its first job); the bounded thread count is the degree of compile parallelism.
/// </summary>
internal sealed class SlangWorkerPool
{
    /// <summary>The pool shared by every <see cref="SlangCompiler"/>.</summary>
    public static SlangWorkerPool Shared { get; } = new(DiagnosticSwitches.CompileThreads);

    private readonly BlockingCollection<(Action<IGlobalSession> Run, Action<Exception> Failed)> _jobs = [];
    private readonly int _threads;
    private int _started;

    private SlangWorkerPool(int threads)
    {
        _threads = Math.Max(1, threads);
    }

    /// <summary>Queues a job; it runs on a worker thread with that worker's global session, and any exception it throws goes to <paramref name="failed"/>.</summary>
    public void Enqueue(Action<IGlobalSession> job, Action<Exception> failed)
    {
        _jobs.Add((job, failed));
        // Start a thread per queued job until the bound is reached, so small workloads do not create idle sessions.
        int started = Volatile.Read(ref _started);
        while (started < _threads)
        {
            if (Interlocked.CompareExchange(ref _started, started + 1, started) == started)
            {
                var thread = new Thread(Work) { IsBackground = true, Name = "slang-compile-" + started };
                thread.Start();
                return;
            }

            started = Volatile.Read(ref _started);
        }
    }

    private void Work()
    {
        IGlobalSession? session = null;
        foreach (var job in _jobs.GetConsumingEnumerable())
        {
            try
            {
                session ??= CreateSession();
                job.Run(session);
            }
            catch (Exception e)
            {
                job.Failed(e);
            }
        }
    }

    private static IGlobalSession CreateSession()
    {
        var result = Slang.CreateGlobalSession(Slang.ApiVersion, out var session);
        if (result.Failed || session is null)
        {
            throw new InvalidOperationException($"Failed to create Slang global session ({result}).");
        }

        return session;
    }
}
