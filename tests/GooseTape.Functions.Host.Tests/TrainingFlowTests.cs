using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace GooseTape.Functions.Host.Tests;

/// <summary>
/// Drives the operations in the same order the durable training feature does, and checks the
/// properties that feature relies on: epochs chain by checkpoint, training improves the
/// network, and an epoch is deterministic enough to retry.
/// </summary>
public sealed class TrainingFlowTests : IClassFixture<FunctionHostFactoryFixture>
{
    private const int HiddenNeurons = 24;
    private const double LearningRate = 0.5d;
    private const int BatchSize = 16;
    private const int Seed = 4242;

    private static readonly int[] Topology = [FunctionHostFactory.SyntheticPixelCount, HiddenNeurons, 10];

    private readonly FunctionHostFactory _factory;

    public TrainingFlowTests(FunctionHostFactoryFixture fixture)
    {
        ArgumentNullException.ThrowIfNull(fixture);

        _factory = fixture.Factory;
    }

    [Fact]
    public async Task AFullRun_ChainsCheckpointsAndImprovesAccuracy()
    {
        var runId = NewRunId();
        var initial = await InitialiseAsync(runId);
        var startingCheckpoint = initial.GetProperty("checkpoint_id").GetString();

        startingCheckpoint.Should().Be($"{runId}-epoch-0");

        var baseline = await EvaluateAsync(startingCheckpoint!);
        var checkpoint = startingCheckpoint!;
        var losses = new List<double>();

        for (var epoch = 1; epoch <= 5; epoch++)
        {
            var trained = await TrainEpochAsync(runId, epoch, checkpoint);

            trained.GetProperty("epoch").GetInt32().Should().Be(epoch);
            trained.GetProperty("checkpoint_id").GetString().Should().Be($"{runId}-epoch-{epoch}");

            losses.Add(trained.GetProperty("mean_loss").GetDouble());
            checkpoint = trained.GetProperty("checkpoint_id").GetString()!;
        }

        var final = await EvaluateAsync(checkpoint);

        losses[^1].Should().BeLessThan(losses[0], "training should reduce loss");
        final.GetProperty("accuracy").GetDouble()
            .Should().BeGreaterThan(baseline.GetProperty("accuracy").GetDouble());
        final.GetProperty("accuracy").GetDouble().Should().BeGreaterThan(0.9d);
    }

    [Fact]
    public async Task ReRunningAnEpoch_ProducesTheSameResult()
    {
        // This is what lets the contract declare train-epoch idempotent, and what makes a
        // Ductape retry of a half-finished epoch safe.
        var runId = NewRunId();
        var initial = await InitialiseAsync(runId);
        var startingCheckpoint = initial.GetProperty("checkpoint_id").GetString()!;

        var first = await TrainEpochAsync(runId, 1, startingCheckpoint);
        var second = await TrainEpochAsync(runId, 1, startingCheckpoint);

        second.GetProperty("mean_loss").GetDouble()
            .Should().Be(first.GetProperty("mean_loss").GetDouble());
        second.GetProperty("accuracy").GetDouble()
            .Should().Be(first.GetProperty("accuracy").GetDouble());
    }

    [Fact]
    public async Task ATrainedNetwork_ClassifiesAnImageOfItsOwnClass()
    {
        var runId = NewRunId();
        var initial = await InitialiseAsync(runId);
        var checkpoint = initial.GetProperty("checkpoint_id").GetString()!;

        for (var epoch = 1; epoch <= 6; epoch++)
        {
            checkpoint = (await TrainEpochAsync(runId, epoch, checkpoint))
                .GetProperty("checkpoint_id").GetString()!;
        }

        // The synthetic generator gives each class its own adjacent pixel pair.
        const int digit = 7;
        var pixels = new double[FunctionHostFactory.SyntheticPixelCount];
        pixels[digit * 2] = 0.9d;
        pixels[(digit * 2) + 1] = 0.9d;

        var response = await InvokeAsync("classify-digit", new { checkpoint_id = checkpoint, pixels });

        response.Status.Should().Be(HttpStatusCode.OK);

        var output = Output(response);
        output.GetProperty("digit").GetInt32().Should().Be(digit);
        output.GetProperty("confidence").GetDouble().Should().BeGreaterThan(0.5d);
        output.GetProperty("distribution").GetArrayLength().Should().Be(10);
    }

    [Fact]
    public async Task ACheckpointFromAnotherRun_IsRefused()
    {
        // The identifier alone proves nothing, so the recorded lineage is what catches this.
        var borrowed = (await InitialiseAsync(NewRunId())).GetProperty("checkpoint_id").GetString()!;

        var response = await InvokeAsync("train-epoch", new
        {
            run_id = NewRunId(),
            epoch = 1,
            from_checkpoint = borrowed,
            learning_rate = LearningRate,
            batch_size = BatchSize,
            seed = Seed,
        });

        response.Status.Should().Be(HttpStatusCode.UnprocessableEntity);
        var error = JsonDocument.Parse(response.Body).RootElement.GetProperty("error");
        error.GetProperty("code").GetString().Should().Be("CHECKPOINT_LINEAGE_MISMATCH");
        error.GetProperty("message").GetString().Should().Contain("belongs to run");
    }

    [Fact]
    public async Task AnEpochThatSkipsItsPredecessor_IsRefused()
    {
        var runId = NewRunId();
        var initial = await InitialiseAsync(runId);
        var epochOne = await TrainEpochAsync(runId, 1, initial.GetProperty("checkpoint_id").GetString()!);

        // Epoch 3 must continue from epoch 2, not epoch 1.
        var response = await InvokeAsync("train-epoch", new
        {
            run_id = runId,
            epoch = 3,
            from_checkpoint = epochOne.GetProperty("checkpoint_id").GetString(),
            learning_rate = LearningRate,
            batch_size = BatchSize,
            seed = Seed,
        });

        response.Status.Should().Be(HttpStatusCode.UnprocessableEntity);
        var error = JsonDocument.Parse(response.Body).RootElement.GetProperty("error");
        error.GetProperty("code").GetString().Should().Be("CHECKPOINT_LINEAGE_MISMATCH");
        error.GetProperty("message").GetString().Should().Contain("must continue from epoch 2");
    }

    [Fact]
    public async Task TrainingFromAnUnknownCheckpoint_Fails()
    {
        var response = await InvokeAsync("train-epoch", new
        {
            run_id = NewRunId(),
            epoch = 1,
            from_checkpoint = "not-a-real-checkpoint",
            learning_rate = LearningRate,
            batch_size = BatchSize,
            seed = Seed,
        });

        response.Status.Should().Be(HttpStatusCode.UnprocessableEntity);
        JsonDocument.Parse(response.Body).RootElement
            .GetProperty("error").GetProperty("code").GetString()
            .Should().Be("CHECKPOINT_NOT_FOUND");
    }

    [Fact]
    public async Task AnOutOfRangeLearningRate_IsRejected()
    {
        var runId = NewRunId();
        var initial = await InitialiseAsync(runId);

        var response = await InvokeAsync("train-epoch", new
        {
            run_id = runId,
            epoch = 1,
            from_checkpoint = initial.GetProperty("checkpoint_id").GetString(),
            learning_rate = 12d,
            batch_size = BatchSize,
            seed = Seed,
        });

        response.Status.Should().Be(HttpStatusCode.UnprocessableEntity);
        JsonDocument.Parse(response.Body).RootElement
            .GetProperty("error").GetProperty("code").GetString()
            .Should().Be("FUNCTION_INPUT_INVALID");
    }

    private async Task<JsonElement> InitialiseAsync(string runId)
    {
        var response = await InvokeAsync("initialize-network", new
        {
            run_id = runId,
            topology = Topology,
            seed = Seed,
        });

        response.Status.Should().Be(HttpStatusCode.OK);

        return Output(response);
    }

    private async Task<JsonElement> TrainEpochAsync(string runId, int epoch, string fromCheckpoint)
    {
        var response = await InvokeAsync("train-epoch", new
        {
            run_id = runId,
            epoch,
            from_checkpoint = fromCheckpoint,
            learning_rate = LearningRate,
            batch_size = BatchSize,
            seed = Seed,
        });

        response.Status.Should().Be(HttpStatusCode.OK);

        return Output(response);
    }

    private async Task<JsonElement> EvaluateAsync(string checkpointId)
    {
        var response = await InvokeAsync("evaluate-network", new { checkpoint_id = checkpointId });

        response.Status.Should().Be(HttpStatusCode.OK);

        return Output(response);
    }

    private static string NewRunId() => $"flow{Guid.NewGuid():N}";

    private async Task<InvocationResponse> InvokeAsync(string operation, object input)
    {
        var invocationId = $"inv-{Guid.NewGuid():N}";
        var body = FunctionHostFactory.BodyFor(operation, input, invocationId);
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
            .ToString(System.Globalization.CultureInfo.InvariantCulture);

        var request = new HttpRequestMessage(
            HttpMethod.Post,
            new Uri(FunctionHostFactory.RouteFor(operation), UriKind.Relative))
        {
            Content = new StringContent(body, Encoding.UTF8),
        };

        request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
        request.Headers.Add("x-ductape-invocation-id", invocationId);
        request.Headers.Add("x-ductape-timestamp", timestamp);
        request.Headers.Add("x-ductape-signature", FunctionHostFactory.Sign(timestamp, body));
        request.Headers.Add(
            "x-ductape-function",
            $"{FunctionHostFactory.FunctionNamespace}.{operation}@{FunctionHostFactory.FunctionVersion}");

        using var client = _factory.CreateClient();
        using var response = await client.SendAsync(request);

        return new InvocationResponse(response.StatusCode, await response.Content.ReadAsStringAsync());
    }

    private static JsonElement Output(InvocationResponse response) =>
        JsonDocument.Parse(response.Body).RootElement.GetProperty("output").Clone();

    private sealed record InvocationResponse(HttpStatusCode Status, string Body);
}
