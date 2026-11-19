using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Feed;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Services.Smoke;

/// <summary>
/// The smoke run of <c>--smoke</c>: lists the devices, installs a tiny package (Whisper tiny, with its Silero VAD companion) from the live Sharplight feed into the
/// model store, decodes a short WAV with the native audio library, loads the package on the CPU (or the provider the run was asked for) and transcribes the WAV once. The first failing
/// step ends the run. It only uses the service interfaces, so tests run it over fakes.
/// </summary>
internal sealed class SmokeRunner
{
    /// <summary>The id prefix of the package the run installs.</summary>
    public const string PackagePrefix = "whisper-tiny";

    private readonly IFeedService _feed;
    private readonly IModelRuntime _runtime;
    private readonly IAudioDecoder _decoder;
    private readonly string _workDirectory;
    private readonly ILogger _logger;
    private readonly string _provider;
    private readonly List<SmokeStep> _steps = [];
    private ComputeDeviceOption? _device;
    private string? _package;
    private string? _packagePath;
    private AudioBuffer? _audio;
    private ILoadedModel? _model;

    /// <param name="feed">The feed to install from.</param>
    /// <param name="runtime">Loads the package.</param>
    /// <param name="decoder">Decodes the WAV file.</param>
    /// <param name="workDirectory">Where the WAV file is written.</param>
    /// <param name="logger">Receives a line per step.</param>
    /// <param name="provider">The execution provider the package is loaded on: <c>cpu</c> (the default), or a GPU provider such as <c>cuda</c> or <c>vulkan</c>
    /// (the run fails when the machine has no device of it).</param>
    public SmokeRunner(IFeedService feed, IModelRuntime runtime, IAudioDecoder decoder, string workDirectory, ILogger logger, string provider = "cpu")
    {
        _provider = provider.Trim().ToLowerInvariant();
        _feed = feed;
        _runtime = runtime;
        _decoder = decoder;
        _workDirectory = workDirectory;
        _logger = logger;
    }

    /// <summary>Runs the steps and returns the report; never throws for a failing step.</summary>
    /// <param name="version">The version of the running build.</param>
    /// <param name="storage">The storage mode and location, for the report.</param>
    /// <param name="cancellationToken">Cancels the run (the running step fails).</param>
    public async Task<SmokeReport> RunAsync(string version, string storage, CancellationToken cancellationToken)
    {
        var watch = Stopwatch.StartNew();
        _ = await StepAsync("devices", ListDevices, cancellationToken)
            && await StepAsync("feed", FindPackageAsync, cancellationToken)
            && await StepAsync("install", InstallAsync, cancellationToken)
            && await StepAsync("decode", Decode, cancellationToken)
            && await StepAsync("load", LoadAsync, cancellationToken)
            && await StepAsync("infer", InferAsync, cancellationToken);
        if (_model is not null)
        {
            await StepAsync("unload", UnloadAsync, CancellationToken.None);
        }

        return new SmokeReport
        {
            Version = version,
            Os = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
            Architecture = System.Runtime.InteropServices.RuntimeInformation.OSArchitecture.ToString(),
            Storage = storage,
            Provider = _provider,
            Devices = [.. _runtime.Devices.Select(device => $"{device.Provider}: {device.Name} ({device.Id})")],
            Package = _package,
            Steps = _steps,
            Seconds = Math.Round(watch.Elapsed.TotalSeconds, 2),
        };
    }

    private async Task<bool> StepAsync(string name, Func<CancellationToken, Task<string>> action, CancellationToken cancellationToken)
    {
        var watch = Stopwatch.StartNew();
        try
        {
            string detail = await action(cancellationToken);
            _steps.Add(new SmokeStep(name, true, detail, Math.Round(watch.Elapsed.TotalSeconds, 2)));
            _logger.LogInformation("Smoke step {Step} passed: {Detail}", name, detail);
            return true;
        }
        catch (Exception error)
        {
            _steps.Add(new SmokeStep(name, false, $"{error.GetType().Name}: {error.Message}", Math.Round(watch.Elapsed.TotalSeconds, 2)));
            _logger.LogError(error, "Smoke step {Step} failed", name);
            return false;
        }
    }

    private Task<string> ListDevices(CancellationToken cancellationToken)
    {
        var devices = _runtime.Devices;
        _ = devices.FirstOrDefault(device => !device.IsGpu) ?? throw new InvalidOperationException("No CPU device is offered.");
        _device = devices.FirstOrDefault(device => string.Equals(device.Provider, _provider, StringComparison.OrdinalIgnoreCase) && !device.IsSplit)
            ?? throw new InvalidOperationException($"No device of the {_provider} provider is offered ({string.Join(", ", devices.Select(device => device.Provider).Distinct())} are).");
        return Task.FromResult($"{devices.Count} device(s): {string.Join("; ", devices.Select(device => device.Display + " via " + device.Provider))}");
    }

    private async Task<string> FindPackageAsync(CancellationToken cancellationToken)
    {
        var listing = await _feed.GetListingAsync(cancellationToken);
        if (listing.Failure is not null)
        {
            throw new InvalidOperationException($"The feed listing failed: {listing.Failure.Message}");
        }

        if (!listing.Verified || listing.FromCache)
        {
            throw new InvalidOperationException($"The listing is not a fresh verified one (verified: {listing.Verified}, from cache: {listing.FromCache}).");
        }

        var package = listing.Packages.FirstOrDefault(candidate => candidate.Id.StartsWith(PackagePrefix, StringComparison.Ordinal))
            ?? throw new InvalidOperationException($"The {_feed.Channel} feed lists no {PackagePrefix} package ({listing.Packages.Count} packages).");
        _package = package.Id;
        return $"{listing.Packages.Count} packages on {_feed.Url} ({_feed.Channel}); using {package.Id}";
    }

    private async Task<string> InstallAsync(CancellationToken cancellationToken)
    {
        var result = await _feed.InstallAsync(_package!, null, cancellationToken);
        _packagePath = result.Requested.PackagePath;
        if (!File.Exists(_packagePath))
        {
            throw new FileNotFoundException("The installed package file is missing.", _packagePath);
        }

        return $"{result.Requested.FeedPackage} {result.Requested.FeedVersion}, {new FileInfo(_packagePath).Length} bytes, {result.Installed.Count} stored";
    }

    private Task<string> Decode(CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(_workDirectory);
        string path = Path.Combine(_workDirectory, "smoke.wav");
        File.WriteAllBytes(path, SmokeAudio.CreateWav());
        _audio = _decoder.Decode(path);
        if (_audio.Samples.Length == 0)
        {
            throw new InvalidDataException("The decoded audio is empty.");
        }

        return Task.FromResult($"{_audio.Samples.Length / _audio.Channels} frames, {_audio.SampleRate} Hz, {_audio.Channels} channel(s)");
    }

    private async Task<string> LoadAsync(CancellationToken cancellationToken)
    {
        var settings = new LoadSettings { DeviceId = _device!.Id, ContextLength = 1024, BatchTokens = 128, Slots = 1 };
        _model = await _runtime.LoadAsync(new ModelLoadRequest("smoke-whisper", _packagePath!, settings, [], Kind: ModelKinds.SpeechRecognition), null, cancellationToken);
        return $"loaded on {_model.Device.Name} ({_model.Device.Id})";
    }

    private async Task<string> InferAsync(CancellationToken cancellationToken)
    {
        using var lease = await _model!.AcquireClientAsync<ISpeechRecognizer>(cancellationToken);
        var recognizer = lease.Client;
        var audio = _audio!.SampleRate == recognizer.SampleRate ? _audio : _audio.Resample(recognizer.SampleRate);
        var transcription = await recognizer.TranscribeAsync(audio, null, cancellationToken);
        // The WAV is a synthetic buzz, not speech: the text is whatever the model makes of it. What counts is that the whole pipeline ran.
        return $"transcribed {audio.Samples.Length / audio.Channels} frames at {recognizer.SampleRate} Hz into {transcription.Segments.Count} segment(s), {transcription.Text.Trim().Length} characters";
    }

    private async Task<string> UnloadAsync(CancellationToken cancellationToken)
    {
        await _runtime.UnloadAsync(_model!.Id);
        return "unloaded";
    }
}
