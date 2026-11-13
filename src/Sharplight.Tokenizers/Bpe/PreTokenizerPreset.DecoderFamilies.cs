namespace Sharplight.Tokenizers.Bpe;

internal sealed partial record PreTokenizerPreset
{
    /// <summary>
    /// The presets of the decoder families added with the generic decoder variants (llama.cpp <c>llama-vocab.cpp</c> naming): Granite and
    /// Command-R split digits one by one before the GPT-2 expression, DBRX / Smaug (SmolLM3) use the Llama-3 expression without merge shortcuts,
    /// GLM-4 the same with merge shortcuts, EXAONE 4 the plain GPT-2 expression and Hunyuan-dense its own three-stage expression.
    /// </summary>
    private static PreTokenizerPreset? ResolveDecoderFamily(string name) => name switch
    {
        "refact" or "command-r" or "starcoder" => SingleDigitGpt2(),
        "falcon" => Falcon(),
        // BLOOM / Poro / GPT-3 Finnish (llama.cpp LLAMA_VOCAB_PRE_TYPE_BLOOM): runs of anything but whitespace and the punctuation of the character class, with an optional leading space.
        "bloom" or "poro" or "gpt3-finnish" => new(new DefaultRegexStages([$@" ?[^(|.,!?…。，、।۔،){DefaultRegexStages.Ws}]+"]), false, false),
        // DBRX / Smaug / OLMo-2 / Granite 4: the Llama-3 expression with ordinary merge application and no default BOS
        // (Granite 4 GGUFs set add_bos_token explicitly; OLMo-2 relies on this default).
        "dbrx" or "smaug-bpe" => new(new RegexFamilySplitter(RegexFamily.Llama, true, 3, false), false, false),
        "glm4" or "glm5" or "chatglm-bpe" => new(new RegexFamilySplitter(RegexFamily.Llama, true, 3, false), true, false),
        "seed-coder" => new(new DefaultRegexStages([@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+"]), false, false),
        "exaone4" => new(new RegexFamilySplitter(RegexFamily.Gpt2, false, 0, false), false, false),
        "hunyuan-dense" => new(new DefaultRegexStages(
        [
            @"\p{N}{1,3}",
            @"[一-龥぀-ゟ゠-ヿ]+",
            @"[!""#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ]), false, false),
        _ => null,
    };

    // Falcon (llama.cpp LLAMA_VOCAB_PRE_TYPE_FALCON): punctuation runs (including the backtick) are cut out first, then the GPT-2 expression
    // without its trailing "\\s+" alternative, then digits in groups of exactly three.
    private static PreTokenizerPreset Falcon()
        => new(new DefaultRegexStages([@"[\p{P}\$\+<=>\^~\|`]+", $@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^{DefaultRegexStages.Ws}\p{{L}}\p{{N}}]+|[{DefaultRegexStages.Ws}]+(?![^{DefaultRegexStages.Ws}])", "[0-9][0-9][0-9]"]), false, false);

    // llama.cpp: digits are split individually first, then the GPT-2 expression (without its trailing "\\s+" alternative).
    private static PreTokenizerPreset SingleDigitGpt2()
        => new(new DefaultRegexStages([@"\p{N}", $@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{{L}}+| ?\p{{N}}+| ?[^{DefaultRegexStages.Ws}\p{{L}}\p{{N}}]+|[{DefaultRegexStages.Ws}]+(?![^{DefaultRegexStages.Ws}])"]), false, false);
}
