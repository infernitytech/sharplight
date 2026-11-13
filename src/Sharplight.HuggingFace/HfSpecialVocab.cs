using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Special-token ids (bos, eos, unk, ...), "add special token" flags and the chat template of a Hugging Face tokenizer,
/// following gguf-py's <c>SpecialVocab</c>: <c>tokenizer_config.json</c> names the tokens, tokenizer.json's added tokens give their ids,
/// the post-processor decides whether BOS/EOS are added, and <c>config.json</c> is the id fallback.
/// </summary>
internal sealed class HfSpecialVocab
{
    private readonly List<string> _kinds = ["bos", "eos", "unk", "sep", "pad", "cls", "mask"];
    private readonly Dictionary<string, int> _ids = new();
    private readonly Dictionary<string, bool> _addFlags = new();
    private bool _passThroughTemplate;

    /// <summary>tokenizer_config entries rewritten by the post-processor logic (they shadow the file's values).</summary>
    private readonly Dictionary<string, string?> _overrides = new();

    private readonly JsonElement? _tokenizerConfig;
    private readonly bool _hasTokenizerConfig;

    private string? _bos;
    private string? _cls;
    private string? _eos;
    private string? _sep;

    private HfSpecialVocab(JsonElement? tokenizerConfig)
    {
        _tokenizerConfig = tokenizerConfig;
        _hasTokenizerConfig = tokenizerConfig is { ValueKind: JsonValueKind.Object } root && root.EnumerateObject().Any();
    }

    /// <summary>Adds the special-token ids, add-token flags and chat template to <paramref name="md"/>.</summary>
    /// <param name="md">The canonical metadata being built.</param>
    /// <param name="dir">The checkpoint directory (chat template files live there).</param>
    /// <param name="config">The checkpoint's <c>config.json</c> (id fallback).</param>
    /// <param name="tok">The root of tokenizer.json.</param>
    /// <param name="tokenizerConfig">The root of tokenizer_config.json, or null when absent.</param>
    /// <param name="addedTokens">The <c>added_tokens</c> entries of tokenizer.json.</param>
    public static void AddTo(IDictionary<string, object> md, string dir, HfConfig config, JsonElement tok, JsonElement? tokenizerConfig, JsonElement[] addedTokens)
    {
        var vocab = new HfSpecialVocab(tokenizerConfig);
        vocab.ReadTokenNames();
        vocab.ApplyPostProcessor(tok);
        vocab.ReadTokenizerConfig(md, dir, addedTokens);
        if (vocab._passThroughTemplate && !Has(vocab.ConfiguredToken("bos_token")))
        {
            // A pass-through template of a tokenizer without a BOS token adds no special tokens, unless tokenizer_config says otherwise.
            vocab._addFlags.TryAdd("bos", false);
            vocab._addFlags.TryAdd("eos", false);
        }

        vocab.ReadConfigJsonIds(config);
        vocab.WriteTo(md);
        // TODO: normalizer flags (NFC/lowercase/strip_accents), clean_up_tokenization_spaces and metaspace prepend scheme
        // are not part of the canonical vocabulary yet, so they are not emitted.
    }

    private static bool Has(string? value) => !string.IsNullOrEmpty(value);

    /// <summary>The token string tokenizer_config names for <paramref name="key"/> (a string or an object with <c>content</c>), honoring overrides.</summary>
    private string? ConfiguredToken(string key)
    {
        if (_overrides.TryGetValue(key, out var overridden))
        {
            return overridden;
        }

        if (_tokenizerConfig is not { ValueKind: JsonValueKind.Object } root || !root.TryGetProperty(key, out var entry))
        {
            return null;
        }

        return entry.ValueKind switch
        {
            JsonValueKind.String => entry.GetString(),
            JsonValueKind.Object when entry.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.String => content.GetString(),
            _ => null,
        };
    }

    /// <summary>Reads bos/cls/eos/sep; like transformers, a missing bos falls back to cls and a missing eos to sep.</summary>
    private void ReadTokenNames()
    {
        _bos = ConfiguredToken("bos_token");
        _cls = ConfiguredToken("cls_token");
        _eos = ConfiguredToken("eos_token");
        _sep = ConfiguredToken("sep_token");
        if (!Has(_bos) && Has(_cls) && _hasTokenizerConfig)
        {
            _overrides["bos_token"] = _bos = _cls;
        }

        if (!Has(_eos) && Has(_sep) && _hasTokenizerConfig)
        {
            _overrides["eos_token"] = _eos = _sep;
        }
    }

    private void ApplyPostProcessor(JsonElement tok)
    {
        if (!tok.TryGetProperty("post_processor", out var postProcessor) || postProcessor.ValueKind != JsonValueKind.Object)
        {
            return;
        }

        // A "Sequence" post-processor lists its steps; otherwise the post-processor is the single step.
        var processors = postProcessor.TryGetProperty("processors", out var steps) && steps.ValueKind == JsonValueKind.Array
            ? steps.EnumerateArray().ToArray()
            : [postProcessor];
        foreach (var processor in processors)
        {
            switch (processor.TypeName())
            {
                case "ByteLevel":
                    _addFlags.TryAdd("bos", false);
                    _addFlags.TryAdd("eos", false);
                    break;
                case "RobertaProcessing":
                    ApplyRobertaProcessing(processor);
                    break;
                case "TemplateProcessing":
                    ApplyTemplateProcessing(processor);
                    break;
            }
        }
    }

    private void ApplyRobertaProcessing(JsonElement processor)
    {
        _addFlags["bos"] = true;
        _addFlags["eos"] = true;
        _addFlags["sep"] = true;
        if (!Has(_cls) && _hasTokenizerConfig && processor.TryGetProperty("cls", out var cls) && cls.ValueKind == JsonValueKind.Array)
        {
            _overrides["cls_token"] = _cls = cls[0].GetString();
        }

        if (!Has(_sep) && _hasTokenizerConfig && processor.TryGetProperty("sep", out var sep) && sep.ValueKind == JsonValueKind.Array)
        {
            _overrides["sep_token"] = _sep = sep[0].GetString();
        }
    }

    /// <summary>
    /// Crude parsing of the single-sequence template to decide whether BOS/EOS are added.
    /// TODO: the pair template (separator tokens) is not interpreted.
    /// </summary>
    private void ApplyTemplateProcessing(JsonElement processor)
    {
        if (!processor.TryGetProperty("single", out var single) || single.ValueKind != JsonValueKind.Array)
        {
            return;
        }

        if (single.GetArrayLength() <= 1)
        {
            // A template that only passes the sequence through adds no special tokens.
            if (SpecialTokenId(single[0]) is null)
            {
                _passThroughTemplate = true;
            }

            return;
        }

        string? first = SpecialTokenId(single[0]);
        string? last = SpecialTokenId(single[single.GetArrayLength() - 1]);
        if (Has(first))
        {
            if (!_hasTokenizerConfig)
            {
                _bos = first;
            }
            else if (first != _bos && first != _cls)
            {
                if (!Has(_bos))
                {
                    _overrides["bos_token"] = _bos = first;
                }

                if (!Has(_cls))
                {
                    _overrides["cls_token"] = _cls = first;
                }
            }

            _addFlags["bos"] = first == _bos || first == _cls;
        }

        if (Has(last))
        {
            if (!_hasTokenizerConfig)
            {
                _eos = last;
            }
            else if (last != _eos)
            {
                // The previous eos becomes end-of-turn (or end-of-message when eot is taken), as llama.cpp does.
                if (!_kinds.Contains("eot"))
                {
                    _kinds.Add("eot");
                    _overrides["eot_token"] = _eos ?? "";
                }
                else if (!_kinds.Contains("eom"))
                {
                    _kinds.Add("eom");
                    _overrides["eom_token"] = _eos ?? "";
                }

                _overrides["eos_token"] = _eos = last;
            }

            _addFlags["eos"] = last == _eos;
        }
    }

    private static string? SpecialTokenId(JsonElement item)
        => item.TryGetProperty("SpecialToken", out var specialToken) && specialToken.TryGetProperty("id", out var id) ? id.GetString() : null;

    private void SetId(string kind, int id)
    {
        if (id < 0)
        {
            throw new InvalidDataException($"invalid value for special token type {kind}: {id}");
        }

        _ids.TryAdd(kind, id);
    }

    /// <summary>Chat template, <c>add_{kind}_token</c> flags and the token ids found by looking the configured token up in tokenizer.json's added tokens.</summary>
    private void ReadTokenizerConfig(IDictionary<string, object> md, string dir, JsonElement[] addedTokens)
    {
        if (!_hasTokenizerConfig)
        {
            return;
        }

        var root = _tokenizerConfig!.Value;
        if (HfChatTemplate.Read(dir, root) is { } template)
        {
            md[CanonicalKeys.ChatTemplate] = template;
        }

        foreach (var kind in _kinds.ToArray())
        {
            if (root.TryGetProperty($"add_{kind}_token", out var flag) && flag.ValueKind is JsonValueKind.True or JsonValueKind.False)
            {
                _addFlags.TryAdd(kind, flag.ValueKind == JsonValueKind.True);
            }

            string? content = ConfiguredToken($"{kind}_token");
            if (!Has(content))
            {
                continue;
            }

            // like llama.cpp: only the added_tokens list is searched
            foreach (var added in addedTokens)
            {
                if (added.GetProperty("content").GetString() == content)
                {
                    SetId(kind, added.GetProperty("id").GetInt32());
                    break;
                }
            }
        }
    }

    /// <summary>config.json fallback for ids (root, then text_config).</summary>
    private void ReadConfigJsonIds(HfConfig config)
    {
        foreach (var kind in _kinds)
        {
            string key = $"{kind}_token_id";
            int? id = null;
            if (config.Root.TryGetProperty(key, out var rootValue) && rootValue.ValueKind == JsonValueKind.Number)
            {
                id = rootValue.GetInt32();
            }
            else if (config.Root.TryGetProperty("text_config", out var textConfig) && textConfig.TryGetProperty(key, out var textValue) && textValue.ValueKind == JsonValueKind.Number)
            {
                id = textValue.GetInt32();
            }

            if (id is { } value)
            {
                SetId(kind, value);
            }
        }
    }

    /// <summary>Writes only the kinds that have a canonical key.</summary>
    private void WriteTo(IDictionary<string, object> md)
    {
        foreach (var (kind, id) in _ids)
        {
            string? key = kind switch
            {
                "bos" => CanonicalKeys.TokenizerBosId,
                "eos" => CanonicalKeys.TokenizerEosId,
                "eot" => CanonicalKeys.TokenizerEotId,
                "eom" => CanonicalKeys.TokenizerEomId,
                "unk" => CanonicalKeys.TokenizerUnkId,
                "pad" => CanonicalKeys.TokenizerPadId,
                "sep" => CanonicalKeys.TokenizerSeparatorId,
                "mask" => CanonicalKeys.TokenizerMaskId,
                "cls" => CanonicalKeys.TokenizerClassifierId,
                _ => null,
            };
            if (key != null)
            {
                md[key] = (uint)id;
            }
        }

        foreach (var (kind, value) in _addFlags)
        {
            string? key = kind switch
            {
                "bos" => CanonicalKeys.TokenizerAddBos,
                "eos" => CanonicalKeys.TokenizerAddEos,
                "sep" => "tokenizer.add_sep_token",
                _ => null,
            };
            if (key != null)
            {
                md[key] = value;
            }
        }
    }
}
