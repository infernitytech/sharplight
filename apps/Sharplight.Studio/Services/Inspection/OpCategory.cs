namespace Sharplight.Studio.Services.Inspection;

/// <summary>Broad families of ops, used to color the nodes of the graph view.</summary>
internal enum OpCategory
{
    /// <summary>Anything else (control flow, sampling, custom ops).</summary>
    Other,

    /// <summary>Matrix products and convolutions.</summary>
    Linear,

    /// <summary>Attention (cached or not).</summary>
    Attention,

    /// <summary>Normalizations.</summary>
    Normalization,

    /// <summary>Elementwise arithmetic and activations.</summary>
    Elementwise,

    /// <summary>Layout and indexing: reshape, transpose, concat, slice, gather.</summary>
    Layout,

    /// <summary>Reads and writes of persistent state (KV caches, recurrent state).</summary>
    State,
}
