using Microsoft.Extensions.DependencyInjection;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.Architectures;

/// <summary>Registers the speech-synthesis implementations of the architecture package with the task layer.</summary>
public static class SpeechSynthesisServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the <see cref="ISpeechSynthesizer"/> factories of the built-in text-to-speech families (OuteTTS with the WavTokenizer decoder).
        /// <c>services.AddSharplight()</c> of the Sharplight.Defaults package calls it.
        /// </summary>
        public IServiceCollection AddSharplightSpeechSynthesis()
        {
            ArgumentNullException.ThrowIfNull(services);
            BuiltinAdapters.RegisterAll();
            return services.AddTaskClientFactory<ISpeechSynthesizer>(ModelKinds.SpeechSynthesis, model => new OuteTtsSynthesizer(model), OuteTtsSynthesizer.Handles);
        }
    }
}
