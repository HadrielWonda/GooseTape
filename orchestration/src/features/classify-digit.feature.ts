import type { IDefineFeatureOptions, IFeatureContext } from '@ductape/sdk';
import {
  neuralNetworkContract,
  type ClassifyDigitOutput,
} from '../contracts/neural-network.contract.js';

/** What a classification request supplies. */
export interface ClassifyRunInput {
  readonly checkpoint_id: string;
  readonly pixels: readonly number[];
}

/**
 * What a classification returns.
 *
 * Only values the feature actually produced at runtime appear here. A derived boolean such as
 * `confident` would be computed while the handler is being recorded, against a placeholder
 * rather than a real probability, so the confidence gate is expressed as a branch instead and
 * the raw probability is handed back for the caller to judge.
 */
export interface ClassifyRunOutput {
  readonly digit: number;
  readonly confidence: number;
}

/** Below this probability the prediction is reported as unconfident rather than accepted. */
const CONFIDENCE_THRESHOLD = 0.6;

/** Ignored by `@ductape/sdk` 0.3.7 for function steps, as in the training feature (#31). */
const CLASSIFY_STEP_OPTIONS = {
  retries: 2,
  retry_interval: 1_000,
  retry_backoff: 'exponential',
  timeout: 120_000,
} as const;

/**
 * Builds the inference feature.
 *
 * Short and single-stepped by design: this is the read path, so it carries none of the
 * checkpointing the training feature needs. It exists as a feature rather than a direct
 * function call so inference inherits the same retries, logging and run history as training.
 *
 * @param product - The Ductape product tag to publish under.
 * @param functionBaseUrl - The C# function host base URL, without a trailing slash.
 * @param sampleRequest - The request shape recorded at definition time.
 * @returns The feature definition, ready to pass to `ductape.feature.define`.
 */
export function classifyDigitFeature(
  product: string,
  functionBaseUrl: string,
  sampleRequest: ClassifyRunInput,
): IDefineFeatureOptions<ClassifyRunInput, ClassifyRunOutput> {
  const contract = neuralNetworkContract(functionBaseUrl);

  return {
    product,
    tag: 'classify-digit',
    name: 'Classify Digit',
    description: 'Classifies one image using a checkpointed network from a training run.',
    controlFlowMode: 'portable',
    recordInput: sampleRequest,
    input: {
      checkpoint_id: { type: 'string', required: true },
      pixels: { type: 'array', required: true },
    },
    handler: async (ctx: IFeatureContext<ClassifyRunInput>): Promise<ClassifyRunOutput> => {
      const functions = ctx.functions.use(contract);

      const prediction = (await ctx.step(
        'classify',
        async () =>
          functions['classify-digit']({
            checkpoint_id: ctx.input.checkpoint_id,
            pixels: ctx.input.pixels,
          }),
        null,
        CLASSIFY_STEP_OPTIONS,
      )) as ClassifyDigitOutput;

      // Recorded as checkpoints: a step must perform a portable operation, and noting the
      // confidence band is not one.
      await ctx.branch(ctx.when.gte(prediction.confidence, CONFIDENCE_THRESHOLD), {
        then: async () => {
          await ctx.checkpoint('confident-prediction', {
            digit: prediction.digit,
            confidence: prediction.confidence,
          });
        },
        else: async () => {
          await ctx.checkpoint('unconfident-prediction', {
            digit: prediction.digit,
            confidence: prediction.confidence,
          });
        },
      });

      return {
        digit: prediction.digit,
        confidence: prediction.confidence,
      };
    },
  };
}
