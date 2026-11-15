using Sharplight.Package;

namespace Sharplight.Studio.Services.Precompile;

/// <summary>
/// Precompiles library packages for every execution provider and device available on this machine (portable code, then device code;
/// identical devices share) and writes the result into the package in place.
/// </summary>
internal interface IPrecompileService
{
    /// <summary>Whether <paramref name="path"/> is being precompiled right now (it must not be loaded meanwhile).</summary>
    public bool IsBusy(string path);

    /// <summary>
    /// Precompiles the package at <paramref name="path"/> and stores the code in it (crash-safe append). The caller makes sure the model
    /// is not resident.
    /// </summary>
    /// <param name="path">The <c>.slm</c> package.</param>
    /// <param name="progress">Receives the stage of each provider and device (on the compiling thread).</param>
    /// <param name="cancellationToken">Cancels before anything is written.</param>
    /// <returns>What was produced, stored and skipped.</returns>
    /// <exception cref="InvalidOperationException">The package is already being precompiled.</exception>
    public Task<PackagePrecompileReport> PrecompileAsync(string path, IProgress<PrecompileProgress> progress, CancellationToken cancellationToken);
}
