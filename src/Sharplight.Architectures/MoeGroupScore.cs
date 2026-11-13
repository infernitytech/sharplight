namespace Sharplight.Architectures;

/// <summary>How group-limited routing ranks an expert group (canonical <c>moe.group_score</c>).</summary>
internal enum MoeGroupScore
{
    /// <summary>"top2_sum": the sum of the group's two best selection scores (DeepSeek-V3, Kimi-K2, Ling 2, GLM-4.5; llama.cpp's only form).</summary>
    TopTwoSum,

    /// <summary>"max": the group's best selection score (DeepSeek-V2 <c>group_limited_greedy</c>).</summary>
    Max,
}
