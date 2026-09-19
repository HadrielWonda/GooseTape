# GooseTape

A from-scratch neural network in C#, trained as a **durable Ductape workflow**.

The network is the one from Milan Jovanović's [*I Built a Neural Network in C# From Scratch*](https://youtu.be/wgNZWnua-90) — dense layers, ReLU, softmax, cross-entropy, backpropagation, no ML libraries. What is different here is everything around it: training does not run as a loop inside a process. Each epoch is a **step in a Ductape feature**, checkpointed and individually retryable. It's designed to resume after a crash, though Ductape can't do that yet on SDK 0.3.7 (see [Durability](#durability-what-is-and-isnt-verified)).

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

- The **epoch loop** uses `ctx.each`, so every epoch is a separate durable step that can fail and retry on its own, and leaves a checkpoint to resume from.
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
- A Ductape workspace at [cloud.ductape.app](https://cloud.ductape.app) and its SDK access key. The product and environment can be created for you, see step 3.

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

Ductape namespaces product tags by workspace, as `<workspace>:<product>`. A product named `goosetape` in a workspace named `goose_tape` has the tag `goose_tape:goosetape`, and that full tag is what `DUCTAPE_PRODUCT` needs. Environment slugs are exactly three characters (`dev`, `prd`, and so on).

### 3. Create the product, if you don't have one

```bash
cd orchestration
npm install
npm run setup
```

This creates the product and environment only if they're missing, so it's safe to re-run. It prints the tag Ductape assigns, so you can check it against `DUCTAPE_PRODUCT`. It runs with the SDK's runtime sync turned off, because with sync on the SDK tries to load the configured product while authenticating, so the call that would create the product fails with `Product/Integration not found`.

### 4. Start the function host

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

### 5. Check everything lines up

```bash
npm run doctor
npm run compile:check
```

`doctor` is read-only. It checks that the key authenticates, the product and environment exist, and the function host is up and serving the contract. `compile:check` runs Ductape's own control-flow validator over each feature handler locally. It needs no credentials and catches a native `if` or `for` before anything is published.

### 6. Publish and train

```bash
npm run publish:features
npm run train -- my-first-run            # synthetic data
npm run train -- my-mnist-run --mnist    # host must be serving MNIST, see Data
```

Publishing records the training feature into 20 steps: `initialise-network`, then `train-epoch-N` plus a checkpoint for each of the 8 epochs, then `evaluate-network`, then the accuracy branch.

Two things `compile:check` can't catch get caught at publish time. Every `ctx.step` must record a portable operation: a function, action or database call, not a plain returned value. And the epoch count is fixed by `recordInput`, so changing it means republishing.

### Durability: what is and isn't verified

**Verified.** Each epoch is its own step, and the epochs chain correctly. The checkpoints a feature run leaves behind show test loss falling on every epoch: 2.3192 → 0.0110 → … → 0.0008 on the synthetic set. So each epoch continues from the one before it rather than training from scratch. Epochs are also deterministic: the same run through the feature and through direct function calls produces the same final loss.

**Not working yet: Ductape's run-management APIs** ([#29](https://github.com/Ductape-LLC/ductape-emails/issues/29)). `feature.execute` runs the feature in the local process. Its step and run results do reach Ductape: they're written through `/integrations/v1/processor/batch-write`, and every run here can be fetched back by ID. But `feature.status`, `history`, `stepDetail`, `resume`, `replay`, `restart`, `cancel`, `signal` and `compare` all call routes under `/integrations/v1/workflow/`, and on 0.3.7 **none of those routes are served**. Each returns Express's default `Cannot GET` or `Cannot POST` page, for real run IDs and made-up ones alike.

The SDK hides this. `status()` and `stepDetail()` turn any 404 into `null`, and `resume()` checks `status()` first and throws `Feature <id> not found`. So a missing endpoint reads as a missing run.

The per-epoch checkpoints on the C# side are exactly what a resume would pick up from, and they're all there. What doesn't work is asking Ductape to do the resuming.

---

## Results

Both runs executed as the durable `train-digit-recogniser` feature through Ductape: 8 epochs, 20 steps, with every invocation HMAC-signed with the real SDK access key. Accuracy is measured on held-out data the network never trained on.

**Real MNIST**: 784 → 128 → 10, learning rate 0.1, batch size 32, 60,000 training and 10,000 test images. The whole run took 3.6 minutes, about 26 s per epoch in a Release build.

| Checkpoint | Test accuracy | Test loss |
|---|---|---|
| untrained | 11.6% | 2.4103 |
| epoch 1 | 95.0% | 0.1676 |
| epoch 2 | 96.5% | 0.1174 |
| epoch 3 | 97.3% | 0.0917 |
| epoch 4 | 97.4% | 0.0879 |
| epoch 5 | 97.5% | 0.0864 |
| epoch 6 | 97.8% | 0.0725 |
| epoch 7 | 97.5% | 0.0791 |
| **epoch 8** | **97.9%** | **0.0710** |

That's in line with what this architecture normally reaches with plain SGD. The dip at epoch 7 is ordinary SGD noise.

**Synthetic**: 20 → 32 → 10. It goes from 8.2% to 100%, and classifies clean prototypes of all ten digits at 99.9% confidence. It only proves the plumbing, because the classes are separable by construction.

Run the host with `-c Release` for MNIST. The matrix code is plain managed loops, and JIT optimisation makes a large difference at this scale.

---

## Data

Out of the box the host uses a **synthetic** dataset (20 pixels, 10 separable classes) so the whole pipeline runs without a download. It proves the plumbing, not the model.

For real MNIST, download the four IDX files into the git-ignored `data/` folder. Google's CVDF mirror is reliable:

```bash
mkdir -p data && cd data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -fLO "https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz"
done
```

then configure the host, either in `appsettings.json` or as `Dataset__Provider=idx` style environment variables:

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

## Ductape SDK issues found along the way

All three are in `@ductape/sdk` 0.3.7. The first two have workarounds in [orchestration/src/config/ductape-client.ts](orchestration/src/config/ductape-client.ts).

- **[#27](https://github.com/Ductape-LLC/ductape-emails/issues/27): the ESM type entry resolves the client to `any`.** `dist/index.d.mts` declares `typeof sdk.default` when `sdk` is already the class. With `skipLibCheck` on, which is the norm, the error is hidden and every client call goes unchecked. Here it hid four mistakes, one of which crashed at runtime. Workaround: type the client from `@ductape/sdk/dist/index`.
- **[#28](https://github.com/Ductape-LLC/ductape-emails/issues/28): `close()` doesn't drain the result queue, and there's no public flush.** `await ductape.close()` followed by `process.exit()` dropped the final execution record in 3 of 3 trials. A plain exit delivers it anyway, but only because an in-flight request happens to keep Node alive. Workaround: `flushPendingWrites()`, which the CLIs call before exiting.
- **[#29](https://github.com/Ductape-LLC/ductape-emails/issues/29): the feature run-management routes aren't served.** Every `/integrations/v1/workflow/` route behind `status`, `history`, `resume`, `replay`, `cancel` and the rest returns Express's default 404 page. The SDK reports that as the run not being found. No workaround: see Durability above.

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
