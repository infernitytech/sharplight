namespace Sharplight.Tokenizers.Bpe;

/// <summary>A resolved <c>tokenizer.pre</c> value: splitter plus the BPE behaviours llama.cpp ties to it.</summary>
internal sealed partial record PreTokenizerPreset(PreTokenizer Splitter, bool IgnoreMerges, bool DefaultAddBos)
{
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
                return new(new DefaultRegexStages([@"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"]), true, false);

            // Flint: digit groups, CJK runs, then Llama-4-style case-aware words (no contraction after digits, punctuation/symbols grouped).
            case "flint":
                return new(new DefaultRegexStages([@"\p{N}{1,3}", @"[一-龥぀-ゟ゠-ヿ]+", @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"]), false, false);

            // DeepSeek LLM / V2: newlines, Latin-script words with an optional leading space, ASCII and full-width punctuation runs,
            // CJK runs and digit runs (llama.cpp LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM). llama.cpp's letter class also lists a few
            // supplementary-plane scripts (Deseret, Osage, Adlam, ...); .NET character classes work on UTF-16 units, so those are left out.
            case "deepseek-llm":
                return new(new DefaultRegexStages([
                    @"[\r\n]",
                    @"\s?[\u0041-\u005A\u0061-\u007A\u00B5\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u01BA\u01BC-\u01BF\u01C4-\u0293\u0295-\u02AF\u0370-\u0373\u0376-\u0377\u037B-\u037D\u037F\u0386\u0388-\u038A\u038C\u038E-\u03A1\u03A3-\u03F5\u03F7-\u0481\u048A-\u052F\u0531-\u0556\u10A0-\u10C5\u13A0-\u13F5\u13F8-\u13FD\u1C90-\u1CBA\u1CBD-\u1CBF\u1D00-\u1D2B\u1D6B-\u1D77\u1D79-\u1D9A\u1E00-\u1F15\u1F18-\u1F1D\u1F20-\u1F45\u1F48-\u1F4D\u1F50-\u1F57\u1F59\u1F5B\u1F5D\u1F5F-\u1F7D\u1F80-\u1FB4\u1FB6-\u1FBC\u1FBE\u1FC2-\u1FC4\u1FC6-\u1FCC\u1FD0-\u1FD3\u1FD6-\u1FDB\u1FE0-\u1FEC\u1FF2-\u1FF4\u1FF6-\u1FFC\u2102\u2107\u210A-\u2113\u2115\u2119-\u211D\u2124\u2126\u2128\u212A-\u212D\u212F-\u2134\u2139\u213C-\u213F\u2145-\u2149\u214E\u2183-\u2184\u2C00-\u2C7B\u2C7E-\u2CE4\u2CEB-\u2CEE\u2CF2-\u2CF3\uA640-\uA66D\uA680-\uA69B\uA722-\uA76F\uA771-\uA787\uA78B-\uA78E\uAB70-\uABBF\uFB00-\uFB06\uFB13-\uFB17\uFF21-\uFF3A\uFF41-\uFF5A]+",
                    @"\s?[\u0021-\u002F\u003A-\u007E\uFF01-\uFF0F\uFF1A-\uFF5E\u2018-\u201F\u3000-\u3002]+",
                    @"\s+$",
                    @"[\u4E00-\u9FA5\u0800-\u4E00\uAC00-\uD7FF]+",
                    @"\p{N}+",
                ]), false, false);

            // DeepSeek-V3 / R1: 1-3 digit groups, CJK / kana runs, then words with a leading punctuation mark or non-letter.
            case "deepseek-v3":
                return new(new DefaultRegexStages([
                    @"\p{N}{1,3}",
                    @"[\u4E00-\u9FA5\u3040-\u309F\u30A0-\u30FF]+",
                    @"[!""#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                ]), false, false);

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
