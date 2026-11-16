namespace Sharplight.Audio.Tests;

public sealed class FuzzTests
{
    [Theory]
    [InlineData("tone_s16.wav")]
    [InlineData("tone.flac")]
    [InlineData("tone.mp3")]
    [InlineData("tone.ogg")]
    [InlineData("tone.opus")]
    public void MutatedFilesNeverCrashTheDecoder(string file)
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        byte[] original = File.ReadAllBytes(Path.Combine(TestSignals.FixtureDirectory, file));
        var decoder = new NativeAudioDecoder();
        var random = new Random(1234);
        for (int iteration = 0; iteration < 150; iteration++)
        {
            byte[] mutated = (byte[])original.Clone();
            int flips = random.Next(1, 9);
            for (int i = 0; i < flips; i++)
            {
                int position = random.Next(4) == 0 ? random.Next(Math.Min(256, mutated.Length)) : random.Next(mutated.Length);
                mutated[position] = (byte)random.Next(256);
            }

            if (random.Next(5) == 0)
            {
                Array.Resize(ref mutated, random.Next(1, mutated.Length));
            }

            try
            {
                AudioBuffer decoded = decoder.Decode(mutated);
                Assert.True(decoded.FrameCount >= 0);
            }
            catch (Exception exception) when (exception is InvalidDataException or IOException or ArgumentException or OutOfMemoryException)
            {
            }
        }
    }
}
