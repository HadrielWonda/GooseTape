import { flushPendingWrites, openSession } from '../config/ductape-client.js';
import type { ClassifyRunOutput } from '../features/classify-digit.feature.js';

/**
 * Classifies one image through the inference feature.
 *
 * Pixels are read from a JSON array argument so the script stays dependency free. Each value
 * must already be normalised to the range zero to one.
 *
 * Usage: `npm run classify -- <checkpoint-id> "[0.0, 0.9, ...]"`
 */
async function main(): Promise<void> {
  const checkpointId = process.argv[2];
  const pixelsArgument = process.argv[3];

  if (checkpointId === undefined || pixelsArgument === undefined) {
    console.error('Usage: npm run classify -- <checkpoint-id> "[0.0, 0.9, ...]"');
    process.exitCode = 1;
    return;
  }

  const pixels = parsePixels(pixelsArgument);
  const { ductape, environment } = openSession();

  const result = await ductape.feature.execute({
    product: environment.product,
    env: environment.env,
    tag: 'classify-digit',
    input: { checkpoint_id: checkpointId, pixels },
  });

  // Deliver the run's records before exiting rather than relying on the SDK; see flushPendingWrites.
  const undelivered = await flushPendingWrites();
  if (undelivered > 0) {
    console.warn(`Warning: ${undelivered} run record(s) were not delivered to Ductape before the deadline.`);
  }

  console.log(`Status: ${result.status}`);
  console.log(`Output: ${JSON.stringify(result.output as ClassifyRunOutput | undefined, null, 2)}`);
}

/**
 * Parses and validates the pixel argument.
 *
 * @param argument - A JSON array of normalised intensities.
 * @returns The parsed pixels.
 * @throws {Error} When the argument is not a JSON array of numbers within zero to one.
 */
function parsePixels(argument: string): number[] {
  let parsed: unknown;

  try {
    parsed = JSON.parse(argument);
  } catch {
    throw new Error('Pixels must be a JSON array, for example "[0.0, 0.9, 0.3]".');
  }

  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error('Pixels must be a non-empty JSON array.');
  }

  const invalid = parsed.findIndex(
    (value) => typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1,
  );

  if (invalid !== -1) {
    throw new Error(`Pixel at index ${invalid} must be a number between 0 and 1.`);
  }

  return parsed as number[];
}

await main();
