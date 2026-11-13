namespace Sharplight.Shaders;

/// <summary>A compiled kernel binary for one target.</summary>
/// <param name="Name">Source file or kernel name, for diagnostics.</param>
/// <param name="EntryPoint">Entry point within the module.</param>
/// <param name="Target">The target the code was generated for.</param>
/// <param name="Code">SPIR-V words, PTX or source text bytes, depending on <paramref name="Target"/>.</param>
/// <param name="CacheKey">Content hash of source, macros and target; identifies the kernel in caches.</param>
public sealed record CompiledKernel(string Name, string EntryPoint, ShaderTarget Target, byte[] Code, string CacheKey);
