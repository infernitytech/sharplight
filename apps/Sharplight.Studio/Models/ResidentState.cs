namespace Sharplight.Studio.Models;

/// <summary>Where a resident model is in its life cycle.</summary>
internal enum ResidentState
{
    /// <summary>Loading onto its device.</summary>
    Loading,

    /// <summary>Loaded and idle.</summary>
    Ready,

    /// <summary>Loaded and generating.</summary>
    Busy,

    /// <summary>Freeing its memory.</summary>
    Unloading,

    /// <summary>Unloaded (idle timeout or evicted for memory); it reloads on its next request.</summary>
    Sleeping,

    /// <summary>The last load failed.</summary>
    Failed,
}
