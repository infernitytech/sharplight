namespace Sharplight.Audio;

/// <summary>
/// The graph contract of the <see cref="Models.ModelKinds.AudioEmbedding"/> kind (CLAP): an audio encoder component (<see cref="Models.ModelComponentRoles.Encoder"/>) with
/// the graph <see cref="AudioGraph"/>, a text encoder component (<see cref="Models.ModelComponentRoles.TextEncoder"/>) with <see cref="TextGraph"/>, a tokenizer and a
/// log-mel feature extractor. Both graphs end in the projection into the shared space.
/// </summary>
public static class AudioEmbedderGraphs
{
    /// <summary>Audio graph: input <see cref="FeaturesInput"/> f32 <c>[frames, mel bins]</c>, outputs <see cref="EmbeddingOutput"/> f32 <c>[1, dimensions]</c> and <see cref="LogitScaleOutput"/> f32 <c>[1]</c> (already exponentiated).</summary>
    public const string AudioGraph = "audio_embed";

    /// <summary>Text graph: inputs <see cref="TokensInput"/> i32 <c>[T]</c> and <see cref="PositionsInput"/> i32 <c>[T]</c>, output <see cref="EmbeddingOutput"/> f32 <c>[1, dimensions]</c> and <see cref="LogitScaleOutput"/>.</summary>
    public const string TextGraph = "text_embed";

    /// <summary>The feature input of the audio graph.</summary>
    public const string FeaturesInput = "features";

    /// <summary>The token input of the text graph.</summary>
    public const string TokensInput = "tokens";

    /// <summary>The position input of the text graph (the model's position ids of the tokens).</summary>
    public const string PositionsInput = "positions";

    /// <summary>The embedding output of both graphs.</summary>
    public const string EmbeddingOutput = "embedding";

    /// <summary>The zero-shot logit scale output.</summary>
    public const string LogitScaleOutput = "logit_scale";

    /// <summary>The audio encoder's configuration key of the sample rate.</summary>
    public const string SampleRateKey = "sampleRate";

    /// <summary>The audio encoder's configuration key of the samples one clip covers (shorter audio is repeated and padded, longer audio is cut or averaged over clips).</summary>
    public const string ClipSamplesKey = "clipSamples";

    /// <summary>The audio encoder's configuration key of the number of feature frames of a clip.</summary>
    public const string FramesKey = "frames";

    /// <summary>The audio encoder's configuration key of the embedding length.</summary>
    public const string DimensionsKey = "dimensions";

    /// <summary>The text encoder's configuration key of the first position id (the padding index + 1 of RoBERTa).</summary>
    public const string PositionOffsetKey = "positionOffset";

    /// <summary>The text encoder's configuration key of the most tokens.</summary>
    public const string MaxTokensKey = "maxTokens";
}
