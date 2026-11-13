using System.Text;

namespace Sharplight.Serving.Tests;

public static class T
{
    public static int[] Prompt(string s) => [.. Encoding.UTF8.GetBytes(s).Select(b => (int)b), FakeBatchDecoder.Marker];
    public static async Task<bool> WaitUntil(Func<bool> cond, int timeoutMs = 5000)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < timeoutMs)
        {
            if (cond()) { return true; }
            await Task.Delay(2);
        }
        return cond();
    }
}
