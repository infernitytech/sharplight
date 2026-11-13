namespace Sharplight.Shaders;

/// <summary>A compiled kernel binary for one target.</summary>
public sealed record CompiledKernel(string Name, string EntryPoint, ShaderTarget Target, byte[] Code, string CacheKey);
