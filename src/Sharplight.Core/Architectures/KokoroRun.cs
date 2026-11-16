namespace Sharplight.Architectures;

/// <summary>The result of one <see cref="KokoroSynthesizer.Run"/>: the symbols fed to the model and what it predicted.</summary>
/// <param name="Ids">The symbol ids including the two padding symbols.</param>
/// <param name="Durations">The rounded duration of every symbol in decoder frames.</param>
/// <param name="RawDurations">The unrounded durations (summed duration sigmoids).</param>
/// <param name="F0">The predicted F0 curve in Hz (two values per frame).</param>
/// <param name="Samples">The waveform.</param>
internal sealed record KokoroRun(int[] Ids, int[] Durations, float[] RawDurations, float[] F0, float[] Samples);
