using Sharplight.Import.PyTorch;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the PyTorch checkpoint format (pickle <c>.bin</c> / <c>.pt</c> / <c>.pth</c> / <c>.ckpt</c> files).</summary>
public static class PyTorchServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the settings of the restricted PyTorch unpickler (<see cref="TorchCheckpointOptions"/>, from Microsoft.Extensions.Options)
        /// that the importers reading pickle checkpoints use: the Hugging Face importer reads <c>pytorch_model*.bin</c> shards with them.
        /// Idempotent; each call adds <paramref name="configure"/>. <c>services.AddHuggingFaceImporter()</c> calls it.
        /// </summary>
        /// <param name="configure">Configures the unpickler, or <see langword="null"/> for the defaults.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddPyTorchImporter(Action<TorchCheckpointOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            var options = services.AddOptions<TorchCheckpointOptions>();
            if (configure is not null)
            {
                options.Configure(configure);
            }

            return services;
        }
    }
}
