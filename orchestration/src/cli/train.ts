import { flushPendingWrites, openSession } from '../config/ductape-client.js';
import {
  DEFAULT_TRAINING_RUN,
  type TrainingRunInput,
  type TrainingRunOutput,
} from '../features/train-digit-recogniser.feature.js';

/**
 * Starts a durable training run and reports how it ended.
 *
 * The run identifier defaults to a timestamped value so repeated runs write to distinct
 * checkpoints rather than overwriting one another. The topology defaults to the synthetic
 * dataset; pass `--mnist` when the function host is serving real MNIST.
 *
 * `--sample=N` trains each epoch on the first N examples only, for quick experiments.
 *
 * Usage: `npm run train -- [runId] [--mnist] [--sample=N]`
 */
const MNIST_TOPOLOGY = [784, 128, 10] as const;
const MNIST_LEARNING_RATE = 0.1;
const MNIST_BATCH_SIZE = 32;

async function main(): Promise<void> {
  const { ductape, environment } = openSession();
  const positional = process.argv.slice(2).filter((argument) => !argument.startsWith('--'));
  const useMnist = process.argv.includes('--mnist');
  const runId = positional[0] ?? `run-${Date.now()}`;
  const sampleSize = readSampleSize(process.argv);

  const input: TrainingRunInput = useMnist
    ? {
        ...DEFAULT_TRAINING_RUN,
        run_id: runId,
        topology: MNIST_TOPOLOGY,
        learning_rate: MNIST_LEARNING_RATE,
        batch_size: MNIST_BATCH_SIZE,
      }
    : { ...DEFAULT_TRAINING_RUN, run_id: runId };
  const run: TrainingRunInput = sampleSize === undefined ? input : { ...input, train_sample_size: sampleSize };

  console.log(`Starting training run ${runId}`);
  console.log(`  epochs:        ${input.epochs.length}`);
  console.log(`  topology:      ${input.topology.join(' -> ')}`);
  console.log(`  learning rate: ${input.learning_rate}`);
  console.log(`  batch size:    ${input.batch_size}`);
  console.log(`  sample size:   ${sampleSize ?? 'full dataset'}`);
  console.log('');

  const result = await ductape.feature.execute({
    product: environment.product,
    env: environment.env,
    tag: 'train-digit-recogniser',
    input: run as unknown as Record<string, unknown>,
  });

  const output = result.output as TrainingRunOutput | undefined;

  console.log(`Status:      ${result.status}`);
  console.log(`Feature run: ${result.feature_id}`);
  console.log(`Duration:    ${Math.round(result.execution_time)}ms`);
  console.log(`Steps:       ${result.completed_steps.length} completed`);

  for (const timing of result.step_timings ?? []) {
    const marker = timing.success ? 'ok  ' : 'FAIL';
    console.log(`  ${marker} ${timing.tag.padEnd(24)} ${Math.round(timing.duration_ms)}ms`);
  }

  console.log(`Output:      ${JSON.stringify(output, null, 2)}`);

  // Deliver the run's records before exiting rather than relying on the SDK; see flushPendingWrites.
  const undelivered = await flushPendingWrites();
  if (undelivered > 0) {
    console.warn(`Warning: ${undelivered} run record(s) were not delivered to Ductape before the deadline.`);
  }

  if (result.status !== 'completed') {
    const where = result.failed_step === undefined ? '' : ` (failed at ${result.failed_step})`;
    console.error(`\nRun did not complete${where}: ${result.error ?? 'no error given'}`);
    console.error(`Every epoch that completed left its checkpoint (${result.completed_steps.length} step(s) completed).`);
    console.error('See the Durability section of the README before relying on feature.resume.');
    process.exitCode = 1;
  }
}

/**
 * Reads `--sample=N` from the command line.
 *
 * @param argv - The process arguments.
 * @returns The sample size, or undefined when the flag is absent.
 * @throws {Error} When the flag is present but not a positive integer.
 */
function readSampleSize(argv: readonly string[]): number | undefined {
  const flag = argv.find((argument) => argument.startsWith('--sample='));

  if (flag === undefined) {
    return undefined;
  }

  const value = Number(flag.slice('--sample='.length));

  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`--sample must be a positive integer, for example --sample=6000. Received ${flag}.`);
  }

  return value;
}

await main();
