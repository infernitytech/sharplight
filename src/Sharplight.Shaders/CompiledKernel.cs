namespace Sharplight.Shaders;

/// <summary>
/// A kernel binary for one target. Kernels returned by <see cref="SlangCompiler"/> may still be compiling in the background:
/// <see cref="Code"/> blocks until the binary is ready, <see cref="IsReady"/> and <see cref="CodeTask"/> do not.
/// Thread-safe.
/// </summary>
public sealed class CompiledKernel
{
    private readonly TaskCompletionSource<byte[]> _code = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly Lock _gate = new();
    private List<Action<byte[]>>? _whenReady = [];
    private byte[]? _ready;

    /// <summary>Creates a kernel whose binary is not available yet; <see cref="Complete"/> or <see cref="Fail"/> finishes it.</summary>
    internal CompiledKernel(string name, string entryPoint, ShaderTarget target, string cacheKey)
    {
        Name = name;
        EntryPoint = entryPoint;
        Target = target;
        CacheKey = cacheKey;
    }

    /// <summary>Creates a kernel from an existing binary.</summary>
    /// <param name="name">Source file or kernel name, for diagnostics.</param>
    /// <param name="entryPoint">Entry point within the module.</param>
    /// <param name="target">The target the code was generated for.</param>
    /// <param name="code">SPIR-V words, PTX or source text bytes, depending on <paramref name="target"/>.</param>
    /// <param name="cacheKey">Content hash of source, macros and target; identifies the kernel in caches.</param>
    public CompiledKernel(string name, string entryPoint, ShaderTarget target, byte[] code, string cacheKey)
        : this(name, entryPoint, target, cacheKey)
    {
        Complete(code);
    }

    /// <summary>Source file or kernel name, for diagnostics.</summary>
    public string Name { get; }

    /// <summary>Entry point within the module.</summary>
    public string EntryPoint { get; }

    /// <summary>The target the code was generated for.</summary>
    public ShaderTarget Target { get; }

    /// <summary>Content hash of source, macros and target; identifies the kernel in caches.</summary>
    public string CacheKey { get; }

    /// <summary>True once the binary is available (or compilation failed).</summary>
    public bool IsReady => _code.Task.IsCompleted;

    /// <summary>Completes with <see cref="Code"/>, or faults with the <see cref="KernelCompilationException"/> of a failed compilation.</summary>
    public Task<byte[]> CodeTask => _code.Task;

    /// <summary>SPIR-V words, PTX or source text bytes, depending on <see cref="Target"/>; waits for a pending compilation.</summary>
    /// <exception cref="KernelCompilationException">Slang rejected the kernel.</exception>
    public byte[] Code => _code.Task.GetAwaiter().GetResult();

    /// <summary>
    /// Runs <paramref name="action"/> with the binary before <see cref="CodeTask"/> completes (immediately when it is ready), so
    /// that anything stored by it is visible to every waiter. Never runs for a failed compilation.
    /// </summary>
    internal void WhenReady(Action<byte[]> action)
    {
        lock (_gate)
        {
            if (_whenReady is not null)
            {
                _whenReady.Add(action);
                return;
            }
        }

        if (_ready is { } code)
        {
            action(code);
        }
    }

    internal void Complete(byte[] code)
    {
        List<Action<byte[]>>? actions;
        lock (_gate)
        {
            _ready = code;
            actions = _whenReady;
            _whenReady = null;
        }

        try
        {
            foreach (var action in actions ?? [])
            {
                action(code);
            }
        }
        catch (Exception e)
        {
            _code.TrySetException(e);
            return;
        }

        _code.TrySetResult(code);
    }

    internal void Fail(Exception error)
    {
        lock (_gate)
        {
            _whenReady = null;
        }

        _code.TrySetException(error);
    }
}
