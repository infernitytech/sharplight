namespace Sharplight.Cli;

/// <summary>Prompt and sampling choices for text generation.</summary>
/// <param name="Prompt">Prompt text; with <see cref="UseChatTemplate"/> it becomes the user message.</param>
/// <param name="MaxNewTokens">Upper bound of generated tokens.</param>
/// <param name="Temperature">Sampling temperature; zero or less selects greedy decoding.</param>
/// <param name="UseChatTemplate">Wrap the prompt as a user message and render the model's chat template.</param>
/// <param name="EnableThinking">Pass <c>enable_thinking</c> to the chat template (for models with a reasoning mode).</param>
/// <param name="SpeculateMtp">Draft tokens per step from the model's multi-token-prediction head (0: off).</param>
/// <param name="SpeculateNgram">Draft tokens per step from prompt lookup (0: off; used when <paramref name="SpeculateMtp"/> is off).</param>
internal sealed record GenerationSettings(
    string Prompt,
    int MaxNewTokens,
    float Temperature,
    bool UseChatTemplate,
    bool EnableThinking,
    int SpeculateMtp = 0,
    int SpeculateNgram = 0)
{
    /// <summary>Draft tokens per step of the selected speculation (0: none).</summary>
    public int DraftTokens => SpeculateMtp > 0 ? SpeculateMtp : SpeculateNgram;
}
