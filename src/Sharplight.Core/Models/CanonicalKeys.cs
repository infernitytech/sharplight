namespace Sharplight.Models;

/// <summary>
/// Canonical metadata vocabulary of a <see cref="ModelSource"/>. Importers translate their format
/// into these keys; adapters, tokenizers and chat formatting read only these.
/// </summary>
/// <remarks>
/// This vocabulary is Sharplight's own and independent of any file format. Hyper-parameters are stored as
/// <c>{architecture}.{key}</c> (see <see cref="ModelSource.Arch{T}(string)"/>) with descriptive keys shared across
/// architectures where the concept is the same: <c>layer_count</c>, <c>hidden_size</c>, <c>ffn_hidden_size</c>,
/// <c>context_length</c>, <c>norm_eps</c>, <c>attention.head_count</c>, <c>attention.kv_head_count</c>,
/// <c>attention.head_dim</c>, <c>rope.dim</c>, <c>rope.base</c>, <c>rope.sections</c>, <c>mtp.layer_count</c>, ...
/// Weight names are hierarchical: <c>token_embedding.weight</c>, <c>layers.{i}.input_norm.weight</c>,
/// <c>layers.{i}.attention.q.weight</c>, <c>layers.{i}.mlp.down.weight</c>, <c>output_norm.weight</c>,
/// <c>lm_head.weight</c>, ... Each architecture adapter documents the exact keys, names and weight semantics
/// it expects; importers (GGUF, Hugging Face, ...) translate into them.
/// <para>Vision towers (image input) have their own canonical vocabulary — hyper-parameters <c>{architecture}.vision.*</c>
/// (<c>layer_count</c>, <c>hidden_size</c>, <c>head_count</c>, <c>patch_size</c>, <c>spatial_merge_size</c>, <c>image_mean</c>, ...)
/// and weights <c>vision.*</c> (<c>vision.patch_embed.weight</c>, <c>vision.layers.{i}.attention.qkv.weight</c>,
/// <c>vision.merger.*</c>, ...) — documented on <see cref="Vision.VisionParameters"/>. They may come in the primary
/// source (Hugging Face checkpoints) or in an auxiliary source with role <see cref="Vision.AuxiliaryRoles.Mmproj"/>
/// (<see cref="ModelImporterRegistry.OpenSet"/>); the architecture adapter treats both alike.</para>
/// </remarks>
public static class CanonicalKeys
{
    public const string Name = "general.name";
    public const string Architecture = "general.architecture";

    public const string TokenizerModel = "tokenizer.model";
    public const string TokenizerPre = "tokenizer.pre";
    public const string TokenizerTokens = "tokenizer.tokens";
    public const string TokenizerTokenTypes = "tokenizer.token_type";
    public const string TokenizerMerges = "tokenizer.merges";
    /// <summary>Per-token scores (float[]) of a SentencePiece-style tokenizer (<c>tokenizer.model</c> = "spm").</summary>
    public const string TokenizerScores = "tokenizer.scores";
    /// <summary>Whether SentencePiece-style tokenizers prepend a space to the text (bool).</summary>
    public const string TokenizerAddSpacePrefix = "tokenizer.add_space_prefix";
    public const string TokenizerBosId = "tokenizer.bos_token_id";
    public const string TokenizerEosId = "tokenizer.eos_token_id";
    public const string TokenizerEotId = "tokenizer.eot_token_id";
    public const string TokenizerEomId = "tokenizer.eom_token_id";
    public const string TokenizerPadId = "tokenizer.padding_token_id";
    public const string TokenizerUnkId = "tokenizer.unknown_token_id";
    public const string TokenizerAddBos = "tokenizer.add_bos_token";
    public const string TokenizerAddEos = "tokenizer.add_eos_token";
    public const string ChatTemplate = "tokenizer.chat_template";
    /// <summary>Separator token id (BERT [SEP], RoBERTa &lt;/s&gt;) used between the two texts of a pair.</summary>
    public const string TokenizerSeparatorId = "tokenizer.separator_token_id";
    /// <summary>Classifier token id (BERT [CLS], RoBERTa &lt;s&gt;).</summary>
    public const string TokenizerClassifierId = "tokenizer.classifier_token_id";
    /// <summary>Mask token id (masked language models).</summary>
    public const string TokenizerMaskId = "tokenizer.mask_token_id";
    /// <summary>
    /// How a text pair is laid out (cross-encoders): "bert" = cls A sep B sep (second text with token type 1),
    /// "roberta" = cls A sep sep B sep (no token types). String.
    /// </summary>
    public const string TokenizerPairStyle = "tokenizer.pair_style";
    /// <summary>WordPiece / text normalization: lowercase the input (bool).</summary>
    public const string TokenizerLowercase = "tokenizer.lowercase";
    /// <summary>WordPiece / text normalization: strip accents (bool; defaults to the lowercase setting like BERT).</summary>
    public const string TokenizerStripAccents = "tokenizer.strip_accents";
    /// <summary>Longest input in tokens the model was trained for (int; embedding / reranking models).</summary>
    public const string TokenizerMaxLength = "tokenizer.max_length";

    /// <summary>Pooling an embedding model was trained with: "mean", "cls", "last" or "none" (string).</summary>
    public const string EmbeddingPooling = "embedding.pooling";
    /// <summary>Whether the embedding model's output is L2-normalized (bool).</summary>
    public const string EmbeddingNormalize = "embedding.normalize";
    /// <summary>
    /// Named prompt prefixes of an embedding model (string[] of "name=prefix" entries, e.g. "query=Instruct: ...\nQuery:"):
    /// prepended to the input text when the caller selects the prompt.
    /// </summary>
    public const string EmbeddingPrompts = "embedding.prompts";
    /// <summary>Default prompt name (string) applied when the caller selects none.</summary>
    public const string EmbeddingDefaultPrompt = "embedding.default_prompt";
    /// <summary>Activation applied to classifier logits to get a score: "sigmoid", "softmax", "none" (string).</summary>
    public const string ClassifierActivation = "classifier.activation";
    /// <summary>Label names of a classifier head (string[]).</summary>
    public const string ClassifierLabels = "classifier.labels";
}
