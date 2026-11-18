namespace Sharplight.Shaders;

/// <summary>
/// Turns the code Slang generates for one target into the form a provider loads, before it is cached (e.g. CUDA C++ into PTX with a
/// runtime compiler). A <see cref="SlangCompiler"/> created with a translator compiles requests for <see cref="Target"/> to
/// <see cref="SourceTarget"/> with Slang and stores the translated code under a cache key that includes <see cref="Id"/>.
/// </summary>
/// <remarks>Implementations must be thread-safe: kernels are translated on the Slang worker threads in parallel.</remarks>
public abstract class KernelTranslator
{
    /// <summary>The target the translator produces (requests for it are translated).</summary>
    public abstract ShaderTarget Target { get; }

    /// <summary>The target Slang generates the input of <see cref="Translate"/> for.</summary>
    public abstract ShaderTarget SourceTarget { get; }

    /// <summary>Identity of the translation (compiler version, options, architecture): part of every cache key of translated kernels.</summary>
    public abstract string Id { get; }

    /// <summary>Translates one kernel.</summary>
    /// <param name="name">Source file or kernel name, for diagnostics.</param>
    /// <param name="entryPoint">Entry point of the kernel.</param>
    /// <param name="code">The code Slang generated for <see cref="SourceTarget"/>.</param>
    /// <returns>The translated code.</returns>
    /// <exception cref="KernelCompilationException">The code could not be translated.</exception>
    public abstract byte[] Translate(string name, string entryPoint, byte[] code);
}
