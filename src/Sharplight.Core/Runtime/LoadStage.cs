namespace Sharplight.Runtime;

/// <summary>Stage of loading a model and compiling it onto devices, as reported through <see cref="LoadProgress"/>.</summary>
public enum LoadStage
{
    /// <summary>Opening the model file(s) and reading metadata through the format importer.</summary>
    Reading,
    /// <summary>Building the architecture's compute graphs and tokenizer from the imported model.</summary>
    Building,
    /// <summary>Planning how a graph is split over the devices (placement); one step per compiled graph.</summary>
    Placing,
    /// <summary>
    /// Compiling a graph for a device: kernels compiled or taken from caches and their pipelines created (GPU providers), or
    /// operations lowered (CPU provider). <see cref="LoadProgress.Cached"/> counts the kernels found in a cache.
    /// </summary>
    Compiling,
    /// <summary>Reading weights and copying them to device memory; progress is in bytes.</summary>
    Uploading,
    /// <summary>A session or batch decoder has been compiled and is ready to run.</summary>
    Ready,
}
