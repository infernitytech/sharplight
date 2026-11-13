using Sharplight.Llm;
using Sharplight.Samples;
using Sharplight.Templating;
using Sharplight.Vision;

namespace Sharplight.Cli;

/// <summary>Generates text for a prompt (optionally with images) and streams it to the console with timing statistics.</summary>
internal sealed class TextGenerator
{
    private const int SamplingSeed = 42;
    private const string NoChatTemplateMessage = "The model has no chat template.";

    private readonly LoadedModel _loaded;
    private readonly GenerationSettings _settings;

    public TextGenerator(LoadedModel loaded, GenerationSettings settings)
    {
        _loaded = loaded;
        _settings = settings;
    }

    public void Run(IReadOnlyList<ImageInput> images, TextWriter output, TextWriter log, CancellationToken cancellationToken)
    {
        using var vision = images.Count > 0 ? _loaded.Model.CreateVisionEncoder() : null;
        var generator = new Generator(_loaded.Session, _loaded.Model.Tokenizer, vision);
        var options = CreateOptions();

        foreach (var token in StartGeneration(generator, images, options, output))
        {
            cancellationToken.ThrowIfCancellationRequested();
            output.Write(token.Text);
        }

        output.WriteLine();
        PrintStatistics(generator, images.Count, log);
    }

    private GenerationOptions CreateOptions()
    {
        float temperature = _settings.Temperature;
        return new GenerationOptions
        {
            MaxNewTokens = _settings.MaxNewTokens,
            Sampling = temperature <= 0 ? SamplingOptions.Greedy : new SamplingOptions { Temperature = temperature, Seed = SamplingSeed },
        };
    }

    private IEnumerable<GeneratedToken> StartGeneration(Generator generator, IReadOnlyList<ImageInput> images, GenerationOptions options, TextWriter output)
    {
        if (images.Count > 0)
        {
            return generator.Generate(RenderImagePrompt(images.Count), images, options);
        }

        if (_settings.UseChatTemplate)
        {
            var promptIds = RenderChatPrompt(output);
            return generator.Generate(promptIds, options);
        }

        output.Write(_settings.Prompt);
        return generator.Generate(_settings.Prompt, options);
    }

    private string RenderImagePrompt(int imageCount)
    {
        // The template turns each image part into the model's placeholder tokens.
        var template = new ChatTemplate(RequireChatTemplate(), bosToken: "", eosToken: "");
        var parts = Enumerable.Range(0, imageCount)
            .Select(_ => ChatContentPart.Image())
            .Append(ChatContentPart.FromText(_settings.Prompt))
            .ToList();
        var context = new Dictionary<string, object?> { ["enable_thinking"] = _settings.EnableThinking };
        return template.Render([new ChatMessage("user", Parts: parts)], addGenerationPrompt: true, context);
    }

    private int[] RenderChatPrompt(TextWriter output)
    {
        var tokenizer = _loaded.Model.Tokenizer;
        var template = new ChatTemplate(RequireChatTemplate(), TokenText(tokenizer.BosId), TokenText(tokenizer.EosId));
        var rendered = template.Render([new ChatMessage("user", _settings.Prompt)], addGenerationPrompt: true);
        output.WriteLine("--- rendered prompt ---");
        output.WriteLine(rendered);
        output.WriteLine("-----------------------");

        // The template emits the BOS token itself.
        return tokenizer.Encode(rendered, addSpecial: false, parseSpecial: true);
    }

    private string? TokenText(int? id)
    {
        return id is { } tokenId ? _loaded.Model.Tokenizer.TokenString(tokenId) : null;
    }

    private string RequireChatTemplate()
    {
        return _loaded.Model.ChatTemplate ?? throw new SampleException(NoChatTemplateMessage);
    }

    private static void PrintStatistics(Generator generator, int imageCount, TextWriter log)
    {
        var stats = generator.LastStats!;
        log.WriteLine(
            $"prompt: {stats.PromptTokens} tok in {stats.PrefillTime.TotalMilliseconds:F0} ms ({stats.PrefillTokensPerSecond:F1} tok/s)"
            + $" | generated: {stats.GeneratedTokens} tok, {stats.DecodeTokensPerSecond:F2} tok/s");
        if (imageCount > 0)
        {
            log.WriteLine($"images: {imageCount} preprocessed and encoded in {generator.LastImageTime.TotalMilliseconds:F0} ms");
        }
    }
}
