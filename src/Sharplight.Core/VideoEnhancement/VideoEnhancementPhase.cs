namespace Sharplight.VideoEnhancement;

/// <summary>What a <see cref="VideoEnhancementUpdate"/> reports progress of.</summary>
public enum VideoEnhancementPhase
{
    /// <summary>Encoding a batch of input frames (resizing, the encoder of a latent model).</summary>
    Encoding,

    /// <summary>Restoring a batch (the network's denoising or restoration passes).</summary>
    Restoring,

    /// <summary>Decoding a batch to pixels and correcting its colours; the update carries the batch's frames.</summary>
    Decoding,

    /// <summary>The clip is finished (<see cref="VideoEnhancementUpdate.Result"/> is set).</summary>
    Completed,
}
