using Sharplight.Architectures;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>The parts of VibeVoice that need no checkpoint: the DPM solver against diffusers' scheduler, the script parser and the transcript parser.</summary>
public class VibeVoiceUnitTests
{
    /// <summary>The timesteps and sigmas of diffusers' <c>DPMSolverMultistepScheduler</c> (squaredcos_cap_v2, v_prediction) for 10 and 20 steps, and a ten-step chain on a synthetic model.</summary>
    [Fact]
    public void Dpm_solver_matches_diffusers()
    {
        var solver = new VibeVoiceDpmSolver(1000, "squaredcos_cap_v2", "v_prediction");
        solver.Begin(10);
        Assert.Equal([999, 899, 799, 699, 599, 500, 400, 300, 200, 100], Enumerable.Range(0, 10).Select(solver.Timestep));
        solver.Begin(20);
        Assert.Equal([999, 949, 899, 849, 799, 749, 699, 649, 599, 549, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50], Enumerable.Range(0, 20).Select(solver.Timestep));

        solver.Begin(10);
        float[] x = [0.5f, -1f, 2f, 0.25f];
        float[] bias = [1f, -1f, 0.5f, 2f];
        for (int i = 0; i < 10; i++)
        {
            var output = new float[4];
            for (int k = 0; k < 4; k++)
            {
                output[k] = (0.3f * x[k]) + (0.1f * bias[k]) + (0.01f * solver.Timestep(i) / 1000f);
            }

            x = solver.Step(output, x);
        }

        float[] expected = [0.17514525f, -0.4921895f, 1.1538472f, -0.10213969f];
        for (int k = 0; k < 4; k++)
        {
            Assert.Equal(expected[k], x[k], 1e-4f);
        }
    }

    /// <summary>Scripts: speaker lines, continuation lines and unlabeled text.</summary>
    [Fact]
    public void Script_parser_reads_turns()
    {
        var turns = VibeVoiceSynthesizer.ParseScript("Speaker 1: Hello there.\nSpeaker 2: Hi!\nHow are you?\nspeaker 1 : Fine.");
        Assert.Equal([new VibeVoiceTurn("1", "Hello there."), new VibeVoiceTurn("2", "Hi! How are you?"), new VibeVoiceTurn("1", "Fine.")], turns);
        Assert.Equal([new VibeVoiceTurn("0", "Just some text.")], VibeVoiceSynthesizer.ParseScript("Just some text."));
        Assert.Empty(VibeVoiceSynthesizer.ParseScript("  \n "));
    }

    /// <summary>The transcript parser yields each segment as soon as its object is complete and survives truncated output.</summary>
    [Fact]
    public void Segment_parser_streams_complete_objects()
    {
        var parser = new VibeVoiceSegmentParser();
        const string Text = "assistant\n[{\"Start\":0.0,\"End\":2.5,\"Speaker\":0,\"Content\":\"Hello, {world}.\"},{\"Start\":2.5,\"End\":4,\"Speaker\":1,\"Content\":\"Hi \\\"there\\\"\"}]";
        Assert.Empty(parser.Feed("assistant\n"));
        Assert.Empty(parser.Feed(Text[..40]));
        var first = parser.Feed(Text[..90]).ToList();
        Assert.Equal([new VibeVoiceSegment(0, 2.5, 0, "Hello, {world}.")], first);
        Assert.Empty(parser.Feed(Text[..110]));
        Assert.Equal([new VibeVoiceSegment(2.5, 4, 1, "Hi \"there\"")], parser.Feed(Text));
        Assert.Empty(parser.Feed(Text));
    }
}
