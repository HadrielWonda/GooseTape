using GooseTape.Functions.Host.Ductape;
using GooseTape.Functions.Host.Operations;
using GooseTape.Functions.Host.Training;
using GooseTape.NeuralNetwork.Checkpoints;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);

// Ductape publishes the access key to its own SDK as DUCTAPE_ACCESS_KEY, so the same variable is
// accepted here rather than forcing a second, host-specific spelling of the same secret.
builder.Configuration.AddEnvironmentVariables();
builder.Configuration["Ductape:AccessKey"] ??= Environment.GetEnvironmentVariable("DUCTAPE_ACCESS_KEY");

builder.Services
    .AddOptions<DuctapeOptions>()
    .Bind(builder.Configuration.GetSection(DuctapeOptions.SectionName))
    .ValidateDataAnnotations()
    .ValidateOnStart();

builder.Services
    .AddOptions<DatasetOptions>()
    .Bind(builder.Configuration.GetSection(DatasetOptions.SectionName))
    .ValidateDataAnnotations()
    .ValidateOnStart();

builder.Services
    .AddOptions<CheckpointOptions>()
    .Bind(builder.Configuration.GetSection(CheckpointOptions.SectionName))
    .ValidateDataAnnotations()
    .ValidateOnStart();

builder.Services.AddSingleton(TimeProvider.System);
builder.Services.AddSingleton<DatasetProvider>();
builder.Services.AddSingleton(NetworkSerializer.Default);

builder.Services.AddSingleton<ICheckpointStore>(services => FileSystemCheckpointStore.Create(
    services.GetRequiredService<IOptions<CheckpointOptions>>().Value.DirectoryPath));

builder.Services.AddSingleton<NetworkCheckpoints>();

builder.Services.AddSingleton<IPortableFunctionOperation, InitializeNetworkOperation>();
builder.Services.AddSingleton<IPortableFunctionOperation, TrainEpochOperation>();
builder.Services.AddSingleton<IPortableFunctionOperation, EvaluateNetworkOperation>();
builder.Services.AddSingleton<IPortableFunctionOperation, ClassifyDigitOperation>();

builder.Services.AddSingleton(services =>
{
    var options = services.GetRequiredService<IOptions<DuctapeOptions>>().Value;

    return PortableFunctionCatalogue.Create(
        options.FunctionNamespace,
        options.FunctionVersion,
        [.. services.GetRequiredService<IEnumerable<IPortableFunctionOperation>>()]);
});

// Makes each invocation single-use for as long as its signature stays valid, so a captured
// request cannot be replayed inside the verifier's window.
builder.Services.AddSingleton(services => InvocationLedger.Create(services.GetRequiredService<TimeProvider>()));

builder.Services.AddSingleton(services => InvocationSignatureVerifier.Create(
    DuctapeAccessKey.Create(services.GetRequiredService<IOptions<DuctapeOptions>>().Value.AccessKey)));

var application = builder.Build();

application.MapPost(PortableFunctionEndpoint.RoutePattern, PortableFunctionEndpoint.HandleAsync);

// Unauthenticated on purpose: it reports only which contract is served, never the key or any data.
application.MapGet("/health", (PortableFunctionCatalogue catalogue) => Results.Json(new
{
    status = "ok",
    function_namespace = catalogue.Namespace,
    function_version = catalogue.Version,
    operations = catalogue.OperationNames,
}));

await application.RunAsync();

/// <summary>
/// Exposes the implicitly generated entry point so the integration tests can host the
/// application in memory.
/// </summary>
public partial class Program;
