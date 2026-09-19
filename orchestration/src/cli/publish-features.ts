import type { IDefineFeatureOptions } from '@ductape/sdk';
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
 * Each feature goes through `feature.define`, which records the handler into its step graph,
 * validates it against the portable control flow rules, and creates or updates the remote
 * feature. `define` fingerprints each definition, so republishing an unchanged feature is a
 * no-op rather than a duplicate.
 *
 * Features are defined one at a time so a failure names the feature it happened in.
 */
async function main(): Promise<void> {
  const { ductape, environment } = openSession();

  console.log(`Publishing features to product ${environment.product} (${environment.env})`);
  console.log(`Function host: ${environment.functionBaseUrl}\n`);

  const definitions = [
    trainDigitRecogniserFeature(environment.product, environment.functionBaseUrl, DEFAULT_TRAINING_RUN),
    classifyDigitFeature(environment.product, environment.functionBaseUrl, {
      checkpoint_id: 'sample-epoch-0',
      pixels: SAMPLE_PIXELS,
    }),
  ];

  for (const definition of definitions) {
    const defined = await ductape.feature.define(
      definition as unknown as IDefineFeatureOptions<unknown, unknown>,
    );

    const steps = defined.schema.steps.map((step) => step.tag);
    console.log(`  published ${defined.tag} with ${steps.length} step(s)`);
    console.log(`    ${steps.join(' -> ')}`);
  }

  console.log(`\nDone. ${definitions.length} feature(s) published.`);
}

await main();
