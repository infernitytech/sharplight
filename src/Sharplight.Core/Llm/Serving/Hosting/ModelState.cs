namespace Sharplight.Llm.Serving.Hosting;

/// <summary>Lifecycle of a registered model.</summary>
public enum ModelState
{
    /// <summary>Registered, holding no device memory.</summary>
    Unloaded,

    /// <summary>A load is in progress.</summary>
    Loading,

    /// <summary>Loaded and serving.</summary>
    Loaded,

    /// <summary>Being drained and released.</summary>
    Unloading,

    /// <summary>The last load attempt failed (see <see cref="ModelInfo.Error"/>); it is retried on the next request.</summary>
    Failed,
}
