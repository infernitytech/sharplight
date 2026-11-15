using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;
using Sharplight.ImageGeneration;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>
/// Registers the task layer with Microsoft.Extensions.DependencyInjection: the <see cref="TaskClientProvider"/>, the client factories,
/// and task clients for configured models. Every task kind is registered the same way: <c>services.AddTaskClient&lt;TClient&gt;(...)</c>
/// (or its named forms such as <see cref="AddSpeechRecognizer"/>) loads the model as a singleton <see cref="TaskModel"/> on first
/// resolution, with <see cref="TaskLoadOptions"/> from Microsoft.Extensions.Options, and creates the client from it; the container
/// disposes the client before the model.
/// </summary>
public static class TaskServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the <see cref="TaskClientProvider"/>, the <see cref="ModelLoader"/> and the built-in text-generation factory (decoder LLMs
        /// and masked-diffusion LMs). Idempotent. <c>services.AddSharplight()</c> of the Sharplight.Defaults package calls it.
        /// </summary>
        public IServiceCollection AddSharplightTasks()
        {
            ArgumentNullException.ThrowIfNull(services);
            services.AddOptions();
            services.AddModelLoader();
            services.TryAddSingleton<TaskClientProvider>();
            services.TryAddEnumerable(ServiceDescriptor.Singleton<ITaskClientFactory, TextGeneratorFactory>());
            return services;
        }

        /// <summary>Adds a task client factory (an implementation of a task kind, e.g. a Whisper-family speech recognizer).</summary>
        /// <param name="factory">The factory.</param>
        public IServiceCollection AddTaskClientFactory(ITaskClientFactory factory)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(factory);
            services.AddSingleton(factory);
            return services;
        }

        /// <summary>Adds a task client factory made of delegates (<see cref="TaskClientFactory{TClient}"/>).</summary>
        /// <typeparam name="TClient">The client interface produced.</typeparam>
        /// <param name="kind">The model kind served (<see cref="ModelKinds"/>; open-ended).</param>
        /// <param name="create">Creates a client for a model of that kind.</param>
        /// <param name="canCreate">Decides whether a model of that kind is served (e.g. by architecture), or <see langword="null"/> for all.</param>
        public IServiceCollection AddTaskClientFactory<TClient>(string kind, Func<TaskModel, TClient> create, Func<TaskModel, bool>? canCreate = null)
            where TClient : class, ITaskClient
            => services.AddTaskClientFactory(new TaskClientFactory<TClient>(kind, create, canCreate));

        /// <summary>
        /// Adds a <typeparamref name="TClient"/> singleton for one model: on first resolution the model is loaded with the
        /// <see cref="TaskLoadOptions"/> named after <paramref name="serviceKey"/> (configured by <paramref name="configure"/> and any other
        /// options configuration), and the client is created by the <see cref="TaskClientProvider"/>. With a <paramref name="serviceKey"/>
        /// the client is a keyed service, so several models of one kind can be registered.
        /// </summary>
        /// <typeparam name="TClient">The client interface (<see cref="ITextGenerator"/>, <see cref="ISpeechRecognizer"/>, ...).</typeparam>
        /// <param name="configure">Configures the model path, devices and compile settings.</param>
        /// <param name="serviceKey">The service key, or <see langword="null"/> for an unkeyed client.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddTaskClient<TClient>(Action<TaskLoadOptions> configure, object? serviceKey = null)
            where TClient : class, ITaskClient
        {
            ArgumentNullException.ThrowIfNull(configure);
            return services.AddTaskClient<TClient>((options, _) => configure(options), serviceKey);
        }

        /// <summary>
        /// Adds a <typeparamref name="TClient"/> singleton for one model (see <see cref="AddTaskClient{TClient}(Action{TaskLoadOptions}, object?)"/>),
        /// configured with access to the container, e.g. to open a device from the <see cref="Sharplight.Runtime.ExecutionProviderCatalog"/>.
        /// </summary>
        /// <typeparam name="TClient">The client interface (<see cref="ITextGenerator"/>, <see cref="ISpeechRecognizer"/>, ...).</typeparam>
        /// <param name="configure">Configures the model path, devices and compile settings from the container's services.</param>
        /// <param name="serviceKey">The service key, or <see langword="null"/> for an unkeyed client.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddTaskClient<TClient>(Action<TaskLoadOptions, IServiceProvider> configure, object? serviceKey = null)
            where TClient : class, ITaskClient
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);
            services.AddSharplightTasks();
            var modelKey = new TaskModelKey(typeof(TClient), serviceKey);
            services.AddOptions<TaskLoadOptions>(modelKey.OptionsName).Configure<IServiceProvider>(configure);
            services.AddKeyedSingleton(modelKey, static (provider, key) =>
            {
                var options = provider.GetRequiredService<IOptionsMonitor<TaskLoadOptions>>().Get(((TaskModelKey)key!).OptionsName);
                return TaskModel.Load(provider.GetRequiredService<ModelLoader>(), options);
            });
            if (serviceKey is null)
            {
                services.AddSingleton(provider => CreateClient<TClient>(provider, modelKey));
            }
            else
            {
                services.AddKeyedSingleton(serviceKey, (provider, _) => CreateClient<TClient>(provider, modelKey));
            }

            return services;
        }

        /// <summary>Adds an <see cref="ITextGenerator"/> for one model (see <see cref="AddTaskClient{TClient}(Action{TaskLoadOptions}, object?)"/>).</summary>
        /// <param name="configure">Configures the model path, devices and compile settings.</param>
        /// <param name="serviceKey">The service key, or <see langword="null"/> for an unkeyed client.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddTextGenerator(Action<TaskLoadOptions> configure, object? serviceKey = null)
            => services.AddTaskClient<ITextGenerator>(configure, serviceKey);

        /// <summary>Adds an <see cref="ISpeechRecognizer"/> for one model (see <see cref="AddTaskClient{TClient}(Action{TaskLoadOptions}, object?)"/>).</summary>
        /// <param name="configure">Configures the model path, devices and compile settings.</param>
        /// <param name="serviceKey">The service key, or <see langword="null"/> for an unkeyed client.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSpeechRecognizer(Action<TaskLoadOptions> configure, object? serviceKey = null)
            => services.AddTaskClient<ISpeechRecognizer>(configure, serviceKey);

        /// <summary>Adds an <see cref="ISpeechSynthesizer"/> for one model (see <see cref="AddTaskClient{TClient}(Action{TaskLoadOptions}, object?)"/>).</summary>
        /// <param name="configure">Configures the model path, devices and compile settings.</param>
        /// <param name="serviceKey">The service key, or <see langword="null"/> for an unkeyed client.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSpeechSynthesizer(Action<TaskLoadOptions> configure, object? serviceKey = null)
            => services.AddTaskClient<ISpeechSynthesizer>(configure, serviceKey);

        /// <summary>Adds an <see cref="IImageGenerator"/> for one model (see <see cref="AddTaskClient{TClient}(Action{TaskLoadOptions}, object?)"/>).</summary>
        /// <param name="configure">Configures the model path, devices and compile settings.</param>
        /// <param name="serviceKey">The service key, or <see langword="null"/> for an unkeyed client.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddImageGenerator(Action<TaskLoadOptions> configure, object? serviceKey = null)
            => services.AddTaskClient<IImageGenerator>(configure, serviceKey);
    }

    private static TClient CreateClient<TClient>(IServiceProvider provider, TaskModelKey modelKey)
        where TClient : class, ITaskClient
        => provider.GetRequiredService<TaskClientProvider>().Create<TClient>(provider.GetRequiredKeyedService<TaskModel>(modelKey));
}
