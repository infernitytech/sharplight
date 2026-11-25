namespace Sharplight.Studio.Services.Library;

/// <summary>What the user is asked to accept before a model is downloaded.</summary>
/// <param name="Key">The model the acceptance is remembered for: a catalog entry id, <c>feed:</c> and the package id, or <c>hub:</c> and the repository.</param>
/// <param name="ModelName">The model's name, for the dialog.</param>
/// <param name="License">The licence name or id.</param>
/// <param name="Notice">The notice text.</param>
/// <param name="Url">A link to the licence or the model page that shows it, or null.</param>
/// <param name="Version">A fingerprint of the licence and notice; a change asks again.</param>
internal sealed record LicenseTerms(string Key, string ModelName, string License, string Notice, string? Url, string Version);
