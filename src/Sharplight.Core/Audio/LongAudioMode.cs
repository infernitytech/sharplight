namespace Sharplight.Audio;

/// <summary>What an audio model that takes fixed-length clips (CLAP: 10 s) does with longer audio.</summary>
public enum LongAudioMode
{
    /// <summary>Only the first clip is used.</summary>
    First,

    /// <summary>Every clip is embedded and the embeddings are averaged.</summary>
    Average,
}
