# GooseTape

A from-scratch neural network in C#, trained as a **durable, resumable Ductape workflow**.

The network is the one from Milan Jovanović's [*I Built a Neural Network in C# From Scratch*](https://youtu.be/wgNZWnua-90) — dense layers, ReLU, softmax, cross-entropy, backpropagation, no ML libraries. What is different here is everything around it: training does not run as a loop inside a process. Each epoch is a **step in a Ductape feature**, checkpointed, individually retryable, and resumable after a crash.

This repo is, in its original spirit, an attempt to break [@snifideezy's Ductape](https://www.ductape.app) by pointing it at a workload it was not designed for.

---

## Why this is an interesting thing to build

Ductape orchestrates backends: APIs, databases, queues, storage, workflows. It is not a numerical computing framework, and nothing in it does matrix arithmetic. So the split is:

| Concern | Where it lives | Why |
|---|---|---|
| Matrix maths, backprop, gradient descent | **C# (.NET 10)** | It is real computation. It belongs in a language with real numerics. |
| Epoch sequencing, retries, checkpoints, resume, run history | **Ductape features (TypeScript)** | This is orchestration, which is exactly what Ductape is for. |
| The boundary between them | **Ductape portable functions over HMAC-signed HTTP** | A first-class Ductape primitive that happens to be language-agnostic. |

Ductape's SDK is Node/TypeScript only — there is no .NET SDK. Portable functions are what make a C# compute tier a legitimate participant rather than a bolt-on.

### The constraint that shaped the design

Ductape compiles a feature by **recording its handler once** into a JSON step graph. In portable mode the SDK statically rejects native control flow:

```
FEATURE_NATIVE_IF     Use ctx.branch(ctx.when.*, { then, else }) instead of a native if statement.
FEATURE_NATIVE_LOOP   Move runtime-sized iteration into ctx.functions instead of a native loop.
```

So a training loop **cannot** be a `for` loop. The options are:

1. `ctx.each` over a fixed collection — expands into one real step per item.
2. Push the iteration into a portable function — runs as one opaque step.

This project uses **both, at different altitudes**:

- The **epoch loop** uses `ctx.each`, so every epoch is a separate durable step that can fail, retry and resume on its own.
- The **mini-batch loop inside an epoch** lives in the C# portable function, where it is one tight numerical loop rather than thousands of workflow steps.

That line is the whole design. Put it too high and you lose durability; too low and you drown the orchestrator in steps.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Ductape Cloud — product, environments, run history      │
└───────────────────────────┬──────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│  orchestration/  (TypeScript, @ductape/sdk)              │
│                                                          │
│   train-digit-recogniser feature                         │
│     step  initialise-network                             │
│     each  epoch 1..N  ->  step train-epoch-N             │
│                           ctx.checkpoint(...)            │
│     step  evaluate-network                               │
│     branch on accuracy vs target                         │
└───────────────────────────┬──────────────────────────────┘
                            │  portable function
                            │  POST /.well-known/ductape/functions/...
                            │  HMAC-SHA256 over `${timestamp}.${body}`
┌───────────────────────────▼──────────────────────────────┐
│  src/GooseTape.Functions.Host  (ASP.NET Core)            │
│    initialize-network · train-epoch                      │
│    evaluate-network   · classify-digit                   │
└───────────────────────────┬──────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│  src/GooseTape.NeuralNetwork  (no dependencies)          │
│    Matrix · DenseLayer · ReLU · Softmax                  │
│    CrossEntropy · backprop · EpochTrainer                │
│    checkpoint store (atomic writes)                      │
└──────────────────────────────────────────────────────────┘
```

### Weights never cross the boundary

A 784→128→10 network is ~100k doubles. Pushing that through a schema-validated function payload once per epoch would dominate the cost of training and bloat every run record.

Instead the two sides exchange **checkpoint identifiers**. The C# host owns weight persistence; the feature passes a string from one epoch to the next:

```
initialise  ->  run-42-epoch-0
epoch 1     ->  reads run-42-epoch-0, writes run-42-epoch-1
epoch 2     ->  reads run-42-epoch-1, writes run-42-epoch-2
```

Checkpoint identifiers are validated against `[A-Za-z0-9_-]{1,128}` at the boundary, because they address files and arrive from outside the process.

### Why an epoch is safe to retry

`FeedForwardNetwork` is immutable — training returns a *new* network. A failed epoch cannot leave torn state behind.

Epochs are also **deterministic**: the shuffle seed is derived as `(runSeed * 397) ^ epochNumber`, deliberately *not* `HashCode.Combine`, which is randomised per process and would make a replay in a fresh process shuffle differently. Same inputs, same epoch, same result — which is what lets `train-epoch` be declared `idempotent: true`.

---

## Layout

```
src/GooseTape.NeuralNetwork/      the network itself, zero dependencies
  Maths/           MatrixShape, Matrix, products, element-wise arithmetic
  Activations/     IActivationFunction, IDifferentiableActivation, ReLU, Softmax
  Losses/          ILossFunction, CrossEntropyLoss
  Layers/          LayerParameters, DenseLayer, LayerCollection, ForwardTrace
  Training/        FeedForwardNetwork, EpochTrainer, value objects
  Data/            MNIST IDX reader, synthetic generator, datasets
  Checkpoints/     snapshots, registries, atomic file store

src/GooseTape.Functions.Host/     ASP.NET Core portable function host
  Ductape/         signature verification, wire protocol, endpoint
  Operations/      the four operations
  Training/        dataset provider, checkpoint service

tests/             67 tests: network maths, gradient checks, HTTP boundary
orchestration/     the Ductape layer
  contracts/       the portable function contract
  features/        the two features
  cli/             publish, train, status, classify, compile-check
```

### Interface segregation, concretely

`SoftmaxActivation` implements `IActivationFunction` but **not** `IDifferentiableActivation`. Its Jacobian is only tractable in combination with cross-entropy, so that pairing lives in `ILossFunction.DeriveOutputDelta`, which returns `prediction - expected`. Softmax therefore cannot be configured as a hidden layer and silently produce wrong gradients — it is not substitutable there, and the type system says so.

---

## Running it

### Prerequisites

- .NET 10 SDK
- Node.js 18+
- A Ductape workspace at [cloud.ductape.app](https://cloud.ductape.app) with a product and at least one environment

### 1. Verify the network

```bash
dotnet test
```

67 tests: 46 over the network itself, 21 over the signed HTTP boundary.

The important ones are in `BackpropagationGradientTests` — they check every analytical gradient against a central-difference numerical estimate, which is what catches a transposed matrix or a dropped activation derivative. `PortableFunctionEndpointTests` covers what the boundary must refuse: unsigned calls, tampered signatures, signatures from the wrong key, expired timestamps, a valid signature redirected at another operation, and checkpoint identifiers that try to escape the store.

### 2. Configure

```bash
cp orchestration/.env.example orchestration/.env
```

Fill in `DUCTAPE_ACCESS_KEY` (workspace → Tokens → SDK Access Key), `DUCTAPE_WORKSPACE`, `DUCTAPE_PUBLISHABLE_KEY`, `DUCTAPE_PRODUCT` and `DUCTAPE_ENV`.

`.env` is git-ignored. The access key is a shared secret and must never be committed.

### 3. Start the function host

The host needs **the same access key** — it verifies the signature Ductape attaches to every call.

```powershell
$env:DUCTAPE_ACCESS_KEY = "<your access key>"
$env:ASPNETCORE_URLS     = "http://127.0.0.1:5207"
dotnet run --project src/GooseTape.Functions.Host
```

```bash
curl http://127.0.0.1:5207/health
```

Plain HTTP is accepted only for localhost. Anywhere else the endpoint must be HTTPS — both the SDK and this host refuse otherwise.

### 4. Check the features compile in portable mode

```bash
cd orchestration
npm install
npm run compile:check
```

This runs Ductape's own control-flow validator locally. It needs no credentials and catches a native `if` or `for` before anything is published.

### 5. Publish and train

```bash
npm run publish:features
npm run train -- my-first-run
npm run status -- <feature-run-id>
```

### Watching it survive a failure

Kill the function host midway through a run. The feature fails on the epoch in flight; the epochs before it stay completed with their checkpoints intact. Restart the host and resume:

```ts
await ductape.feature.resume({ product, env, feature_id, from_checkpoint: 'epoch-4-complete' });
```

Training picks up from epoch 5, not from scratch. That is the entire point of the exercise.

---

## Data

Out of the box the host uses a **synthetic** dataset (20 pixels, 10 separable classes) so the whole pipeline runs without a download. It proves the plumbing, not the model.

For real MNIST, download the four IDX files, then:

```json
"Dataset": {
  "Provider": "idx",
  "TrainingImagesPath": "data/train-images-idx3-ubyte.gz",
  "TrainingLabelsPath": "data/train-labels-idx1-ubyte.gz",
  "TestImagesPath":     "data/t10k-images-idx3-ubyte.gz",
  "TestLabelsPath":     "data/t10k-labels-idx1-ubyte.gz"
}
```

Gzipped files are read directly. The topology's first entry must equal the dataset's pixel count — 784 for MNIST, 20 for synthetic — and the host rejects a mismatch rather than training something meaningless.

---

## Security

- The access key is read from the environment, held only as HMAC key bytes, and its `ToString()` returns `[redacted access key]` so it cannot reach a log through interpolation.
- Signatures are compared in fixed time (`CryptographicOperations.FixedTimeEquals`). An early-returning comparison leaks how much of a forged signature was correct.
- Requests older than five minutes are rejected, matching the SDK's replay window.
- The signature is verified against the **raw body before it is parsed**, so malformed payloads never reach a deserialiser.
- Route segments and the `x-ductape-function` header are both checked against the signed body, so a validly signed request cannot be redirected at another operation in flight.
- Every input field is re-validated at the boundary, even though Ductape validates against the contract schema first — this host is reachable by anything holding the key.

## Conventions

C# follows the house guidelines: file-scoped namespaces, nullable enabled, warnings as errors, XML docs on every public member, value objects instead of bare primitives, first-class collections, small behaviour-rich types, and structured logging via source-generated `LoggerMessage` delegates. TypeScript follows the same principles with `strict` plus `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes`.
