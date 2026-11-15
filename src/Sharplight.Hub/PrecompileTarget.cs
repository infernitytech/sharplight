using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.Hub;

/// <summary>
/// What a pulled package is precompiled for, and how: every device of the given execution providers, through their two-stage
/// precompilation contract (see <see cref="PackagePrecompiler"/>).
/// </summary>
/// <param name="Providers">The execution providers; unavailable ones and providers without precompilable code are skipped.</param>
/// <param name="Options">Context length, batch size and tiers; use the values the model will run with.</param>
/// <param name="Progress">Receives the stage of each provider and device, or <see langword="null"/>.</param>
public sealed record PrecompileTarget(IReadOnlyList<IExecutionProvider> Providers, PrecompileOptions Options, IProgress<PrecompileProgress>? Progress = null);
