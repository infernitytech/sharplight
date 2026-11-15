namespace Sharplight.Tokenizers.Bpe;

internal sealed partial record PreTokenizerPreset
{
    // Contractions spelled out case-insensitively (llama.cpp's adaptation of "(?i:'s|'t|...)").
    private const string Contractions = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)";

    // The Qwen 2 expression (single digits) and the Llama 3 expression (1-3 digit groups).
    private const string Qwen2Pattern = Contractions + @"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    // DeepSeek-V3 / Hunyuan / Hy-v4 / JoyAI: 1-3 digit groups, CJK and kana runs, then words with a leading punctuation mark or non-letter.
    private static readonly string[] s_deepSeek3Stages =
    [
        @"\p{N}{1,3}",
        @"[一-龥぀-ゟ゠-ヿ]+",
        @"[!""#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ];

    // Tekken (Mistral) word pieces: case-aware letters and marks.
    private const string TekkenWords = @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*";

    /// <summary>
    /// The remaining byte-level BPE presets of llama.cpp's <c>llama-vocab.cpp</c> (<c>tokenizer.ggml.pre</c> values written by
    /// <c>convert_hf_to_gguf.py</c>), each with llama.cpp's split expressions and its <c>ignore_merges</c> / <c>add_bos</c> defaults. Presets that
    /// do not byte-encode (<c>gemma4</c>-style ▁ vocabularies such as <c>sarvam-moe</c>, <c>whitespace</c>, <c>soprano</c>) are not byte-level
    /// BPE and are not listed.
    /// </summary>
    private static PreTokenizerPreset? ResolveExtended(string name) => name switch
    {
        // MiniCPM 5: 1-3 digit groups first, then the Llama 3 word expression with digit runs.
        "minicpm5" => Stages(true, false, @"\p{N}{1,3}", Contractions + @"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        "deepseek-coder" => Stages(false, false, @"[\r\n]", @"\s?\p{L}+", @"\s?\p{P}+", @"[一-龥ࠀ-一가-퟿]+", @"\p{N}"),
        "hy_v4" or "joyai-llm" => Stages(false, false, s_deepSeek3Stages),
        "spark2_5" => Stages(false, false, @"\p{N}{1,3}", @"[一-龥぀-ゟ゠-ヿ]+",
            @"[!""#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+|[\r\n]|\s+(?!\S)|\s+", @"\p{N}"),
        "youtu" => Stages(true, false, @"[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+",
            @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+" + Contractions + @"?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*" + Contractions
            + @"?|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        // Jais 2: the Llama 3 expression with a cascade of whitespace run lengths.
        "jais-2" => Stages(false, false, Contractions + @"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s{512}(?!\S)|\s{256}(?!\S)|\s{128}(?!\S)|\s{64}(?!\S)|\s{32}(?!\S)|\s{16}(?!\S)|\s{8}(?!\S)|\s{4}(?!\S)|\s{1,2}(?!\S)|\s{1}"),
        "jina-v1-en" or "jina-v2-code" or "roberta-bpe" => new(new RegexFamilySplitter(RegexFamily.Gpt2, false, 0, false), false, false),
        "poro-chat" => Resolve("poro"),
        "viking" => Stages(false, false, $@" ?[^(|.,!?…。，、।۔،){DefaultRegexStages.Ws}]+", @"\p{N}"),
        "nemotron-3.5-embed" => Stages(true, true, TekkenWords + @"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        // Tekken with numbers split into right-aligned groups of three digits.
        "tekken-num3" => Stages(true, true, TekkenWords + @"|\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        "exaone-moe" => Stages(false, false, Contractions + @"|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]?|\s*[\r\n]|\s+(?!\S)|\s+"),
        "chameleon" => Stages(false, true, "<sentinel:[0-9]+>", "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z", "([\\t\\n]|    |  )", @"\p{N}", @"[\p{P}!-/:-@\[-`{-~]",
            $@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^{DefaultRegexStages.Ws}\p{{L}}\p{{N}}]+|[{DefaultRegexStages.Ws}]+(?![^{DefaultRegexStages.Ws}])"),
        "minerva-7b" or "mellum2" => SingleDigitGpt2(),
        "granite-embed-multi-97m" => Stages(true, false, Gpt4oPattern),
        // Tiny Aya / Command A MoE: right-aligned groups of three digits, then the o200k-style words.
        "tiny_aya" or "cohere2moe" => Stages(false, false, @"\d{1,3}(?=(?:\d{3})*\b)", Gpt4oPattern),
        // SuperBPE: digit runs, then a zero-width split before every complete group of three digits from the right.
        "superbpe" => Stages(false, false, @"\p{N}+", @"(?=(\d{3})+(?!\d))"),
        // Qwen 2 without the English contractions (Turkish apostrophe suffixes stay attached).
        "ufakzeka" => Stages(false, false, @"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        "grok-2" => Stages(false, false, Qwen2Pattern),
        // AFMoE: digit runs grouped in threes from the right (llama.cpp's custom \p{AFMoE_digits}), CJK and other Asian scripts, then the
        // DeepSeek-V3 word expression.
        "afmoe" => Stages(false, false, @"\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))", "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+",
            @"[!""#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        "laguna" => Stages(false, false, @"[^\n]+|[\n]+", Qwen2Pattern),
        "k2-horizon" => Stages(false, false, Contractions + @"|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|‌|‍)+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
        _ => null,
    };

    private static PreTokenizerPreset Stages(bool ignoreMerges, bool defaultAddBos, params string[] stages)
        => new(new DefaultRegexStages(stages), ignoreMerges, defaultAddBos);
}
