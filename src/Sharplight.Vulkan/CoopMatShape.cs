namespace Sharplight.Vulkan;

/// <summary>One supported cooperative-matrix configuration: A[M,K]·B[K,N] + C[M,N].</summary>
internal readonly record struct CoopMatShape(uint M, uint N, uint K, string A, string B, string C, string Result)
{
    private static string Short(string t) => t.Replace("Float", "f").Replace("Sint", "i").Replace("Uint", "u").Replace("KHR", "").Replace("Bfloat", "bf");
    public override string ToString() => $"{M}x{N}x{K}:{Short(A)}*{Short(B)}+{Short(C)}->{Short(Result)}";
    public bool Is(uint m, uint n, uint k, string a, string c) => M == m && N == n && K == k && A == a && B == a && C == c && Result == c;
}
