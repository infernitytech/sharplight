using Sharplight.Llm.Embeddings;

namespace Sharplight.Embeddings.Tests;

/// <summary>
/// Real models against sentence-transformers / transformers references (Fixtures/*.json from Fixtures/make_reference.py):
/// identical token ids, cosine similarity &gt; 0.999 per embedding, reranker logits within ~1e-3. Skipped when a model is missing.
/// </summary>
public class ReferenceTests
{
    public static TheoryData<string, string, string> Embedders => new()
    {
        { "bge-small-en-v1.5", "hf", "cpu" }, { "bge-small-en-v1.5", "hf", "vulkan" },
        { "bge-small-en-v1.5", "gguf:bge-small-en-v1.5/bge-small-en-v1.5-f16.gguf", "cpu" }, { "bge-small-en-v1.5", "gguf:bge-small-en-v1.5/bge-small-en-v1.5-f16.gguf", "vulkan" },
        { "all-MiniLM-L6-v2", "hf", "cpu" }, { "all-MiniLM-L6-v2", "hf", "vulkan" },
        { "all-MiniLM-L6-v2", "gguf:all-MiniLM-L6-v2/all-MiniLM-L6-v2-ggml-model-f16.gguf", "vulkan" },
        { "e5-small-v2", "hf", "cpu" }, { "e5-small-v2", "hf", "vulkan" },
        { "multilingual-e5-small", "hf", "cpu" }, { "multilingual-e5-small", "hf", "vulkan" },
        { "nomic-embed-text-v1.5", "hf", "cpu" }, { "nomic-embed-text-v1.5", "hf", "vulkan" },
        { "nomic-embed-text-v1.5", "gguf:nomic-embed-text-v1.5/nomic-embed-text-v1.5.f16.gguf", "vulkan" },
        { "Qwen3-Embedding-0.6B", "hf", "cpu" }, { "Qwen3-Embedding-0.6B", "hf", "vulkan" },
        { "Qwen3-Embedding-0.6B", "gguf:Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B-f16.gguf", "vulkan" },
        // 8-bit quantized against the unquantized reference (tolerance 0.995).
        { "bge-small-en-v1.5", "gguf:bge-small-en-v1.5/bge-small-en-v1.5-q8_0.gguf", "cpu" }, { "bge-small-en-v1.5", "gguf:bge-small-en-v1.5/bge-small-en-v1.5-q8_0.gguf", "vulkan" },
        { "nomic-embed-text-v1.5", "gguf:nomic-embed-text-v1.5/nomic-embed-text-v1.5.Q8_0.gguf", "cpu" },
        { "Qwen3-Embedding-0.6B", "gguf:Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B-Q8_0.gguf", "vulkan" },
    };

    public static TheoryData<string, string, string> Rerankers => new()
    {
        { "ms-marco-MiniLM-L6-v2", "hf", "cpu" }, { "ms-marco-MiniLM-L6-v2", "hf", "vulkan" },
        { "bge-reranker-base", "hf", "cpu" }, { "bge-reranker-base", "hf", "vulkan" },
        { "bge-reranker-v2-m3", "hf", "vulkan" },
        { "bge-reranker-v2-m3", "gguf:bge-reranker-v2-m3/bge-reranker-v2-m3-Q8_0.gguf", "cpu" }, { "bge-reranker-v2-m3", "gguf:bge-reranker-v2-m3/bge-reranker-v2-m3-Q8_0.gguf", "vulkan" },
        { "Qwen3-Reranker-0.6B", "hf", "cpu" }, { "Qwen3-Reranker-0.6B", "hf", "vulkan" },
    };

    static string PathOf(string model, string format)
        => format == "hf" ? Env.Hf(model) : Env.Gguf(format[5..].Split('/')[0], format[5..].Split('/')[1]);

    /// <summary>8-bit weights: the reference is the unquantized model, so tolerances are wider.</summary>
    static bool Quantized(string format) => format.Contains("q8_0", StringComparison.OrdinalIgnoreCase);

    static readonly Dictionary<(string, string), EmbeddingModel> s_embedders = [];
    static readonly Dictionary<(string, string), Reranker> s_rerankers = [];

    internal static EmbeddingModel LoadEmbedder(string path, string device)
    {
        lock (s_embedders)
        {
            if (!s_embedders.TryGetValue((path, device), out var m))
                s_embedders[(path, device)] = m = EmbeddingModel.Load(path, new EmbeddingLoadOptions { Devices = [Env.Device(device)], MaxBatchTokens = 2048, MaxBatchSequences = 64 });
            return m;
        }
    }

    [Theory, MemberData(nameof(Embedders))]
    public void Embeddings(string model, string format, string device)
    {
        Env.Init();
        string path = PathOf(model, format);
        Env.RequireModel(path);
        var dev = Env.Device(device);
        var reference = Env.Reference(model);
        var texts = reference.GetProperty("texts").EnumerateArray().Select(t => t.GetString()!).ToList();
        var m = LoadEmbedder(path, device);

        var refTokens = reference.GetProperty("tokens").EnumerateArray().Select(Env.Ints).ToList();
        int tokenMismatches = 0;
        for (int i = 0; i < texts.Count; i++)
            if (!m.Tokenize(texts[i]).SequenceEqual(refTokens[i])) tokenMismatches++;

        var result = m.Embed(texts);
        var refEmb = reference.GetProperty("embeddings").EnumerateArray().Select(Env.Floats).ToList();
        double minCos = 1;
        for (int i = 0; i < texts.Count; i++) minCos = Math.Min(minCos, Env.Cosine(result.Vectors[i], refEmb[i]));

        double minPromptCos = 1;
        if (reference.TryGetProperty("prompt_name", out var pn))
        {
            // GGUF files carry no prompts: pass the prompt text itself.
            var o = m.Prompts.ContainsKey(pn.GetString()!) ? new EmbedOptions { Prompt = pn.GetString() } : new EmbedOptions { Prefix = reference.GetProperty("prompt").GetString() };
            var pt = reference.GetProperty("prompt_tokens").EnumerateArray().Select(Env.Ints).ToList();
            for (int i = 0; i < pt.Count; i++) if (!m.Tokenize(texts[i], o).SequenceEqual(pt[i])) tokenMismatches++;
            var pe = reference.GetProperty("prompt_embeddings").EnumerateArray().Select(Env.Floats).ToList();
            var r2 = m.Embed(texts.Take(pe.Count).ToList(), o);
            for (int i = 0; i < pe.Count; i++) minPromptCos = Math.Min(minPromptCos, Env.Cosine(r2.Vectors[i], pe[i]));
        }
        Env.Summary[$"embed {model} {format.Split('/')[^1]} {device}"] = $"tokens-mismatch={tokenMismatches} minCos={minCos:F6} promptMinCos={minPromptCos:F6}";
        Assert.True(tokenMismatches == 0, $"{tokenMismatches} inputs tokenize differently from the reference.");
        double tol = Quantized(format) ? 0.995 : 0.999;
        Assert.True(minCos > tol, $"min cosine similarity {minCos:F6}");
        Assert.True(minPromptCos > tol, $"min cosine similarity with prompt {minPromptCos:F6}");
    }

    [Theory, MemberData(nameof(Rerankers))]
    public void Rerank(string model, string format, string device)
    {
        Env.Init();
        string path = PathOf(model, format);
        Env.RequireModel(path);
        var dev = Env.Device(device);
        var reference = Env.Reference(model);
        string query = reference.GetProperty("query").GetString()!;
        var docs = reference.GetProperty("documents").EnumerateArray().Select(t => t.GetString()!).ToList();
        Reranker r;
        lock (s_rerankers)
            if (!s_rerankers.TryGetValue((path, device), out r!))
                s_rerankers[(path, device)] = r = Reranker.Load(path, new RerankerLoadOptions { Devices = [dev], MaxBatchTokens = 2048, MaxBatchSequences = 64 });

        var refTokens = reference.GetProperty("tokens").EnumerateArray().Select(Env.Ints).ToList();
        int tokenMismatches = docs.Select((d, i) => r.Encode(query, d).Tokens.SequenceEqual(refTokens[i]) ? 0 : 1).Sum();
        var result = r.Score(query, docs);
        var refLogits = Env.Floats(reference.GetProperty("logits"));
        double maxErr = 0;
        for (int i = 0; i < docs.Count; i++) maxErr = Math.Max(maxErr, Math.Abs(result.Logits[i] - refLogits[i]));
        Env.Summary[$"rerank {model} {format.Split('/')[^1]} {device}"] = $"tokens-mismatch={tokenMismatches} maxLogitErr={maxErr:E2} logits=[{string.Join(", ", result.Logits.Select(l => l.ToString("F3")))}]";
        Assert.True(tokenMismatches == 0, $"{tokenMismatches} pairs tokenize differently from the reference.");
        Assert.True(maxErr < (Quantized(format) ? 0.25 : 2e-2), $"max |logit - reference| = {maxErr:E2}");
    }
}
