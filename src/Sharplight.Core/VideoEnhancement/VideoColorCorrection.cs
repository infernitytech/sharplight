namespace Sharplight.VideoEnhancement;

/// <summary>How the colours of the enhanced frames are tied back to the input (<see cref="VideoEnhancementOptions.ColorCorrection"/>).</summary>
public enum VideoColorCorrection
{
    /// <summary>The model's output as it is.</summary>
    None,

    /// <summary>
    /// Wavelet colour transfer: the output's high frequencies over the input's low frequencies (five levels of a dilated 3 × 3 binomial blur
    /// with edge-replicating borders), which keeps the input's colours and brightness and the model's detail.
    /// </summary>
    Wavelet,

    /// <summary>Adaptive instance normalization: each frame and channel of the output gets the mean and standard deviation of the input's.</summary>
    AdaIn,
}
