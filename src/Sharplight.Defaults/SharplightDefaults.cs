using Microsoft.Extensions.DependencyInjection;

namespace Sharplight;

/// <summary>
/// Convenience for hosts without a service container of their own: builds a <see cref="ServiceProvider"/> with
/// <c>services.AddSharplight()</c> (<see cref="SharplightServiceCollectionExtensions"/>) plus whatever <c>configure</c> adds.
/// </summary>
/// <example>
/// <code>
/// using var services = SharplightDefaults.CreateServiceProvider(s => s.AddTextGenerator(o =>
/// {
///     o.ModelPath = "qwen3-4b-q4_k_m.gguf";
///     o.Devices = [slot];
/// }));
/// var generator = services.GetRequiredService&lt;ITextGenerator&gt;();
/// var result = await generator.GenerateAsync("The capital of France is");
/// </code>
/// </example>
public static class SharplightDefaults
{
    /// <summary>Builds a service provider with Sharplight's defaults and the given registrations.</summary>
    /// <param name="configure">Adds further services (task clients, options), or <see langword="null"/>.</param>
    /// <returns>The provider; dispose it to dispose the task clients and models it created.</returns>
    public static ServiceProvider CreateServiceProvider(Action<IServiceCollection>? configure = null)
    {
        var services = new ServiceCollection().AddSharplight();
        configure?.Invoke(services);
        return services.BuildServiceProvider();
    }
}
