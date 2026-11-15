using Sharplight.Cpu;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the CPU execution provider.</summary>
public static class CpuServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the CPU execution provider (<see cref="CpuExecutionProvider"/>). Register it after GPU providers so that discovery lists
        /// the GPUs first. Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddCpuProvider() => services.AddExecutionProvider<CpuExecutionProvider>();
    }
}
