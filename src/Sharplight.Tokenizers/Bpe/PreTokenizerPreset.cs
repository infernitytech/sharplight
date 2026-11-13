namespace Sharplight.Tokenizers.Bpe;

/// <summary>A resolved <c>tokenizer.pre</c> value: splitter plus the BPE behaviours llama.cpp ties to it.</summary>
internal sealed partial record PreTokenizerPreset(PreTokenizer Splitter, bool IgnoreMerges, bool DefaultAddBos)
{
    // Case-aware word pieces with optional contraction suffix and 1-3 digit groups (OpenAI o200k, Llama 4).
    private const string Gpt4oPattern = @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    public static PreTokenizerPreset Resolve(string? name)
    {
        switch (name)
        {
            case null or "" or "default":
                return new(DefaultRegexStages.Instance, false, false);

            case "qwen35":
                return new(new RegexFamilySplitter(RegexFamily.Llama, caseInsensitiveContractions: true, maxDigits: 1, marks: true), false, false);

            case "qwen2" or "deepseek-r1-qwen" or "kormo" or "f2llmv2" or "megrez" or "stablelm2" or "hunyuan" or "solar-open":
                return new(new RegexFamilySplitter(RegexFamily.Llama, true, 1, false), false, false);

            case "llama3" or "llama-v3" or "llama-bpe" or "falcon3" or "falcon-h1" or "midm-2.0" or "jina-v5-nano":
                return new(new RegexFamilySplitter(RegexFamily.Llama, true, 3, false), true, true);

            // Mistral "Tekken" (Pixtral, Nemotron): case-aware word pieces, single digits, ignore_merges.
            case "pixtral" or "tekken":
                return new(new DefaultRegexStages([@"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"]), true, false);

            // Llama 4 / Muse: the Tekken-style case-aware words with optional contraction suffix and 1-3 digit groups.
            case "llama4":
                return new(new DefaultRegexStages([Gpt4oPattern]), true, false);

            // o200k (gpt-oss, MiniMax-M2, Kanana 2): the same word pattern as Llama 4, with the merges applied (llama.cpp GPT4O).
            case "gpt-4o" or "minimax-m2" or "kanana2" or "talkie":
                return new(new DefaultRegexStages([Gpt4oPattern]), false, false);

            // Ling / Ring 2.0 (llama.cpp BAILINGMOE): single digits; tokenizer.json's possessive quantifiers become atomic groups.
            case "bailingmoe" or "bailingmoe2" or "llada-moe":
                return new(new DefaultRegexStages([@"'(?i:[sdmt]|ll|ve|re)|(?>[^\r\n\p{L}\p{N}]?)\p{L}+|\p{N}| ?(?>[^\s\p{L}\p{N}]+)[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"]), false, false);

            // Flint: digit groups, CJK runs, then Llama-4-style case-aware words (no contraction after digits, punctuation/symbols grouped).
            case "flint":
                return new(new DefaultRegexStages([@"\p{N}{1,3}", @"[一-龥぀-ゟ゠-ヿ]+", @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"]), false, false);

            case "lfm2":
                return new(new RegexFamilySplitter(RegexFamily.Llama, true, 3, false), true, true);

            case "smollm" or "codeshell" or "exaone" or "minerva":
                return SingleDigitGpt2();

            case "gpt-2" or "gpt2" or "phi-2" or "jina-es" or "jina-de" or "gigachat" or "jina-v2-es" or "jina-v2-de"
                or "a.x-4.0" or "mellum" or "modern-bert" or "mpt" or "olmo" or "jais" or "trillion" or "granite-docling":
                return new(new RegexFamilySplitter(RegexFamily.Gpt2, false, 0, false), false, false);

            default:
                return ResolveDecoderFamily(name) ?? throw new NotSupportedException($"Pre-tokenizer '{name}' is not supported.");
        }
    }
}
