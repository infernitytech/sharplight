using Sharplight.Llm;

namespace Sharplight.Speech;

/// <summary>Resolves the token sampling of a speech-synthesis request against a model's own settings.</summary>
internal static class SpeechSynthesisOptionsExtensions
{
    extension(SpeechSynthesisOptions? options)
    {
        /// <summary>
        /// The sampling to use: the request's <see cref="SpeechSynthesisOptions.Sampling"/> when it changes anything but the seed, else the model's
        /// <paramref name="modelDefaults"/> with the request's seed.
        /// </summary>
        /// <param name="modelDefaults">The model's own sampling settings.</param>
        public SamplingOptions SamplingOr(SamplingOptions modelDefaults)
        {
            ArgumentNullException.ThrowIfNull(modelDefaults);
            var requested = options?.Sampling;
            if (requested is null || requested with { Seed = null } == new SamplingOptions())
            {
                return modelDefaults with { Seed = requested?.Seed ?? modelDefaults.Seed };
            }

            return requested;
        }
    }
}
