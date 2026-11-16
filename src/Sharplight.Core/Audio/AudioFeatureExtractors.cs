using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>Creates the <see cref="IAudioFeatureExtractor"/> a model's feature-extractor component describes (its <c>type</c> configuration).</summary>
internal static class AudioFeatureExtractors
{
    /// <summary>The extractor of <paramref name="model"/>'s <see cref="ModelComponentRoles.FeatureExtractor"/> component.</summary>
    /// <param name="model">The loaded model.</param>
    /// <exception cref="KeyNotFoundException">The model has no feature extractor.</exception>
    /// <exception cref="NotSupportedException">The extractor type is unknown.</exception>
    public static IAudioFeatureExtractor Create(TaskModel model)
    {
        var component = model.Description.Component(ModelComponentRoles.FeatureExtractor);
        return component.ConfigValue("type", "") switch
        {
            LogMelSpectrogram.Type => new LogMelSpectrogram(component, model.ReadAttachment(component.Role, ModelComponentAttachments.MelFilters)),
            WaveformFeatures.Type => new WaveformFeatures(component),
            var other => throw new NotSupportedException(
                $"Audio feature extractor type '{other}' is not supported (supported: {LogMelSpectrogram.Type}, {WaveformFeatures.Type})."),
        };
    }
}
