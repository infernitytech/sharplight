using System.Diagnostics;
using Sharplight.Tokenizers;
using Sharplight.Vision;

namespace Sharplight.Llm;

/// <summary>
/// Autoregressive text generation on top of an <see cref="InferenceSession"/>. With a <see cref="VisionEncoder"/> (and a
/// session created on the multimodal decoder graph) prompts may contain images. Generations are lazy iterators that run the
/// model while enumerated; the session keeps the context, so successive calls continue the conversation. Not thread-safe.
/// </summary>
/// <param name="session">The session to run on; the generator does not own it.</param>
/// <param name="tokenizer">Tokenizer of the same model.</param>
/// <param name="vision">Vision encoder for prompts with images; null for text-only use.</param>
public sealed class Generator(InferenceSession session, ITokenizer tokenizer, VisionEncoder? vision = null)
{
    /// <summary>Timings and token counts of the last generation (set when its enumeration ends or is abandoned).</summary>
    public GenerationStats? LastStats { get; private set; }

    /// <summary>Continues the session's context with <paramref name="prompt"/> and streams generated text.</summary>
    public IEnumerable<GeneratedToken> Generate(string prompt, GenerationOptions options, bool parseSpecial = true)
        => Generate(tokenizer.Encode(prompt, addSpecial: session.Position == 0, parseSpecial: parseSpecial), options);

    /// <summary>Continues the session's context with already tokenized <paramref name="promptTokens"/> and streams generated text.</summary>
    public IEnumerable<GeneratedToken> Generate(int[] promptTokens, GenerationOptions options)
        => Generate(promptTokens, () => session.Process(promptTokens), options);

    /// <summary>Time spent preprocessing and encoding the images of the last multimodal generation.</summary>
    public TimeSpan LastImageTime { get; private set; }

    /// <summary>
    /// Continues the context with <paramref name="prompt"/> containing one image placeholder (the chat template's
    /// <c>&lt;|image_pad|&gt;</c> token text) per entry of <paramref name="images"/>, in order, and streams generated text.
    /// The architecture's multimodal support expands the placeholders (see <see cref="IMultimodalArchitecture"/>).
    /// </summary>
    public IEnumerable<GeneratedToken> Generate(string prompt, IReadOnlyList<ImageInput> images, GenerationOptions options, bool parseSpecial = true)
    {
        if (images.Count == 0)
        {
            return Generate(prompt, options, parseSpecial);
        }

        var enc = vision ?? throw new InvalidOperationException("Images need a Generator created with a VisionEncoder (LlmModel.CreateVisionEncoder).");
        var arch = MultimodalRegistry.Find(enc.Definition.Architecture)
            ?? throw new NotSupportedException($"Architecture '{enc.Definition.Architecture}' has no multimodal prompt support.");
        var tokens = tokenizer.Encode(prompt, addSpecial: session.Position == 0, parseSpecial: parseSpecial);
        var sw = Stopwatch.StartNew();
        var pre = images.Select(enc.Preprocess).ToList();
        var mm = arch.Assemble(new MultimodalContext(enc.Definition, LookupToken), tokens, pre, session.RopePosition);
        var embeddings = pre.Select(enc.Encode).ToList();
        LastImageTime = sw.Elapsed;
        var textTokens = tokens;
        return Generate(textTokens, () => session.Process(mm, embeddings), options, mm.Length);
    }

    private int? LookupToken(string text)
    {
        var ids = tokenizer.Encode(text, addSpecial: false, parseSpecial: true);
        return ids.Length == 1 ? ids[0] : null;
    }

    private IEnumerable<GeneratedToken> Generate(int[] promptTokens, Func<float[]> prefillStep, GenerationOptions options, int? promptRows = null)
    {
        var sampler = new Sampler(options.Sampling);
        foreach (var t in promptTokens)
        {
            sampler.Accept(t);
        }

        var decoder = new StreamingDecoder(tokenizer);
        var text = new System.Text.StringBuilder();

        var sw = Stopwatch.StartNew();
        var logits = prefillStep();
        var prefill = sw.Elapsed;
        sw.Restart();
        int generated = 0;
        try
        {
            for (int i = 0; i < options.MaxNewTokens; i++)
            {
                int id = sampler.Sample(logits);
                sampler.Accept(id);
                generated++;
                if (options.StopAtEog && tokenizer.IsEndOfGeneration(id))
                {
                    break;
                }

                var piece = decoder.Push(id);
                text.Append(piece);
                yield return new GeneratedToken(id, piece);
                if (options.StopStrings.Any(s => text.ToString().EndsWith(s, StringComparison.Ordinal)))
                {
                    break;
                }

                if (i + 1 < options.MaxNewTokens)
                {
                    logits = session.Process([id]);
                }
            }
            var tail = decoder.Flush();
            if (tail.Length > 0)
            {
                yield return new GeneratedToken(-1, tail);
            }
        }
        finally
        {
            LastStats = new GenerationStats(promptRows ?? promptTokens.Length, generated, prefill, sw.Elapsed);
        }
    }
}
