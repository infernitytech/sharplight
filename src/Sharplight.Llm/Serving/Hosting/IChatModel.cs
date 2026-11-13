using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>A loaded model that serves chat/completion requests.</summary>
public interface IChatModel
{
    /// <summary>The scheduler serving the model.</summary>
    public InferenceEngine Engine { get; }
    /// <summary>The chat layer over <see cref="Engine"/>.</summary>
    public ModelService Service { get; }
}
