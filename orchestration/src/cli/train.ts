import { openSession } from '../config/ductape-client.js';
import {
  DEFAULT_TRAINING_RUN,
  type TrainingRunInput,
  type TrainingRunOutput,
} from '../features/train-digit-recogniser.feature.js';

/**
 * Starts a durable training run and reports how it ended.
 *
 * The run identifier defaults to a timestamped value so repeated runs write to distinct
 * checkpoints rather than overwriting one another.
 *
 * Usage: `npm run train -- [runId]`
 */
async function main(): Promise<void> {
  const { ductape, environment } = openSession();
  const runId = process.argv[2] ?? `run-${Date.now()}`;

  const input: TrainingRunInput = { ...DEFAULT_TRAINING_RUN, run_id: runId };

  console.log(`Starting training run ${runId}`);
  console.log(`  epochs:        ${input.epochs.length}`);
  console.log(`  topology:      ${input.topology.join(' -> ')}`);
  console.log(`  learning rate: ${input.learning_rate}`);
  console.log(`  batch size:    ${input.batch_size}`);
  console.log('');

  const result = await ductape.feature.execute<TrainingRunOutput>({
    product: environment.product,
    env: environment.env,
    tag: 'train-digit-recogniser',
    input: input as unknown as Record<string, unknown>,
  });

  console.log(`Status: ${result.status}`);
  console.log(`Feature run: ${result.feature_id ?? 'unknown'}`);
  console.log(`Output: ${JSON.stringify(result.output, null, 2)}`);

  if (result.status !== 'completed') {
    console.error(
      `\nRun did not complete. Inspect it with:\n  npm run status -- ${result.feature_id ?? '<feature-run-id>'}` +
        '\nand resume from the last good epoch with ductape.feature.resume.',
    );
    process.exitCode = 1;
  }
}

await main();
