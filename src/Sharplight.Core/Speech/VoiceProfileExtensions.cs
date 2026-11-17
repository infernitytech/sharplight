namespace Sharplight.Speech;

/// <summary>Checks of a <see cref="VoiceProfile"/> against the model that is asked to speak with it.</summary>
internal static class VoiceProfileExtensions
{
    extension(VoiceProfile profile)
    {
        /// <summary>The tensor of that name.</summary>
        /// <param name="name">The tensor's name.</param>
        /// <exception cref="InvalidDataException">The voice has no such tensor.</exception>
        public VoiceTensor Tensor(string name)
            => profile.Tensors.TryGetValue(name, out var tensor) ? tensor : throw new InvalidDataException($"The voice '{profile.Name}' has no '{name}' data.");

        /// <summary>Throws unless the voice belongs to the architecture family.</summary>
        /// <param name="architecture">The model's family.</param>
        /// <exception cref="ArgumentException">The voice was made for another model family.</exception>
        public void RequireArchitecture(string architecture)
        {
            if (!string.Equals(profile.Architecture, architecture, StringComparison.Ordinal))
            {
                throw new ArgumentException($"The voice '{profile.Name}' was made for {profile.Architecture} models, not {architecture}.", nameof(profile));
            }
        }
    }
}
