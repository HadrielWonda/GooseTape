import { openSession } from '../config/ductape-client.js';
import {
  trainDigitRecogniserFeature,
  DEFAULT_TRAINING_RUN,
} from '../features/train-digit-recogniser.feature.js';
import { classifyDigitFeature } from '../features/classify-digit.feature.js';

/** A sample image recorded at definition time so the classify handler has an input shape. */
const SAMPLE_PIXELS = Array.from({ length: 20 }, () => 0);

/**
 * Publishes both features to the configured Ductape product.
 *
 * `publishMany` verifies each write by reading it back, and rolls the whole set back if any one
 * write fails, so the product is never left holding half of a definition change.
 */
async function main(): Promise<void> {
  const { ductape, environment } = openSession();

  console.log(`Publishing features to product ${environment.product} (${environment.env})`);
  console.log(`Function host: ${environment.functionBaseUrl}`);

  const definitions = [
    trainDigitRecogniserFeature(environment.product, environment.functionBaseUrl, DEFAULT_TRAINING_RUN),
    classifyDigitFeature(environment.product, environment.functionBaseUrl, {
      checkpoint_id: 'sample-epoch-0',
      pixels: SAMPLE_PIXELS,
    }),
  ];

  const published = await ductape.feature.publishMany(environment.product, definitions);

  for (const feature of published.features) {
    console.log(`  published ${feature.tag} with ${feature.steps.length} step(s)`);
  }

  console.log(`\nDone. ${published.features.length} feature(s) published.`);
}

await main();
