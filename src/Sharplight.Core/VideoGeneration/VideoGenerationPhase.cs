namespace Sharplight.VideoGeneration;

/// <summary>What a <see cref="VideoGenerationUpdate"/> reports progress of.</summary>
public enum VideoGenerationPhase
{
    /// <summary>Encoding the prompts and the start image.</summary>
    Conditioning,

    /// <summary>Denoising steps.</summary>
    Denoising,

    /// <summary>Decoding latent frames to pixels.</summary>
    Decoding,

    /// <summary>The video is finished (<see cref="VideoGenerationUpdate.Result"/> is set).</summary>
    Completed,
}
