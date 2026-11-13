using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>A loaded model that serves chat/completion requests.</summary>
public interface IChatModel
{
    public InferenceEngine Engine { get; }
    public ModelService Service { get; }
}
