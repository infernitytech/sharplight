namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>Configuration keys of the SeedVR2 model description (model-level unless noted).</summary>
internal static class SeedVrConfigKeys
{
    /// <summary>The timestep of the one-step sampler (σ · 1000; 1000 starts from pure noise).</summary>
    public const string Timestep = "timestep";

    /// <summary>The default colour correction (<c>wavelet</c>, <c>adain</c> or <c>none</c>).</summary>
    public const string ColorCorrection = "colorCorrection";

    /// <summary>The default enlargement when a request gives no size.</summary>
    public const string DefaultScale = "defaultScale";

    /// <summary>Sizes the graphs take are multiples of this many pixels.</summary>
    public const string SizeMultiple = "sizeMultiple";

    /// <summary>Clips hold 1 + k · this many frames.</summary>
    public const string FrameMultiple = "frameMultiple";

    /// <summary>The longest batch a whole-clip request is restored in (longer clips and frame sources are cut into batches of this many frames).</summary>
    public const string MaxFramesPerBatch = "maxFramesPerBatch";

    /// <summary>VAE component: spatial compression.</summary>
    public const string SpatialDownscale = "spatialDownscale";

    /// <summary>VAE component: temporal compression.</summary>
    public const string TemporalDownscale = "temporalDownscale";
}
