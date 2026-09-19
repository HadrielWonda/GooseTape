import type { IDefineFeatureOptions, IFeatureContext } from '@ductape/sdk';
import {
  neuralNetworkContract,
  type InitializeNetworkOutput,
  type EvaluateNetworkOutput,
  type TrainEpochOutput,
} from '../contracts/neural-network.contract.js';

/**
 * What a training run is given when it starts.
 *
 * `epochs` is an explicit list rather than a count. Ductape compiles a feature by recording its
 * handler once into a step graph, so the number of epochs has to be known at definition time;
 * `ctx.each` over this list expands into one durable, individually retryable step per epoch.
 * A runtime-sized loop would have to live inside a portable function instead, where it would
 * lose exactly the per-epoch durability this feature exists to provide.
 */
export interface TrainingRunInput {
  readonly run_id: string;
  readonly epochs: readonly number[];
  readonly topology: readonly number[];
  readonly seed: number;
  readonly learning_rate: number;
  readonly batch_size: number;
  readonly target_accuracy: number;
  readonly train_sample_size?: number;
}

/** What a completed training run reports. */
export interface TrainingRunOutput {
  readonly run_id: string;
  readonly final_checkpoint: string;
  readonly epochs_completed: number;
}

/** Retry policy for an epoch. Epochs are deterministic, so a retry recomputes the same result. */
const EPOCH_STEP_OPTIONS = {
  retries: 2,
  retry_interval: 5_000,
  retry_backoff: 'exponential',
  timeout: 600_000,
  critical: true,
} as const;

const SETUP_STEP_OPTIONS = {
  retries: 3,
  retry_interval: 2_000,
  retry_backoff: 'exponential',
  timeout: 120_000,
  critical: true,
} as const;

/** The default schedule, used when the CLI is run without overrides. */
export const DEFAULT_TRAINING_RUN: TrainingRunInput = {
  run_id: 'goosetape-run',
  epochs: [1, 2, 3, 4, 5, 6, 7, 8],
  topology: [20, 32, 10],
  seed: 20_260_917,
  learning_rate: 0.5,
  batch_size: 16,
  target_accuracy: 0.9,
};

/**
 * Builds the durable training feature.
 *
 * Every epoch is a checkpointed step. If the function host dies midway through a run, the
 * completed epochs stay completed and `feature.resume` picks up from the last checkpoint rather
 * than retraining from scratch.
 *
 * @param product - The Ductape product tag to publish under.
 * @param functionBaseUrl - The C# function host base URL, without a trailing slash.
 * @param sampleRun - The run shape recorded at definition time, which fixes the epoch count.
 * @returns The feature definition, ready to pass to `ductape.feature.define`.
 */
export function trainDigitRecogniserFeature(
  product: string,
  functionBaseUrl: string,
  sampleRun: TrainingRunInput = DEFAULT_TRAINING_RUN,
): IDefineFeatureOptions<TrainingRunInput, TrainingRunOutput> {
  const contract = neuralNetworkContract(functionBaseUrl);

  return {
    product,
    tag: 'train-digit-recogniser',
    name: 'Train Digit Recogniser',
    description:
      'Trains the from-scratch C# neural network one durable epoch at a time, checkpointing after each.',
    controlFlowMode: 'portable',
    recordInput: sampleRun,
    input: {
      run_id: { type: 'string', required: true },
      epochs: { type: 'array', required: true },
      topology: { type: 'array', required: true },
      seed: { type: 'number', required: true },
      learning_rate: { type: 'number', required: true },
      batch_size: { type: 'number', required: true },
      target_accuracy: { type: 'number', required: true },
      train_sample_size: { type: 'number', required: false },
    },
    // No rollback strategy is set, and no step declares a rollback handler. A completed epoch is
    // an immutable checkpoint, not a side effect to be undone: unwinding a failed run would throw
    // away exactly the work resuming is meant to preserve.
    options: {
      timeout: 7_200_000,
    },
    handler: async (ctx: IFeatureContext<TrainingRunInput>): Promise<TrainingRunOutput> => {
      const functions = ctx.functions.use(contract);

      const initialised = (await ctx.step(
        'initialise-network',
        async () =>
          functions['initialize-network']({
            run_id: ctx.input.run_id,
            topology: ctx.input.topology,
            seed: ctx.input.seed,
          }),
        null,
        SETUP_STEP_OPTIONS,
      )) as InitializeNetworkOutput;

      ctx.setState('starting_checkpoint', initialised.checkpoint_id);

      let latestCheckpoint: string = initialised.checkpoint_id;

      await ctx.each(ctx.sampleInput.epochs, async (epoch) => {
        const trained = (await ctx.step(
          `train-epoch-${epoch}`,
          async () =>
            functions['train-epoch']({
              run_id: ctx.input.run_id,
              epoch,
              from_checkpoint: latestCheckpoint,
              learning_rate: ctx.input.learning_rate,
              batch_size: ctx.input.batch_size,
              seed: ctx.input.seed,
            }),
          null,
          EPOCH_STEP_OPTIONS,
        )) as TrainEpochOutput;

        latestCheckpoint = trained.checkpoint_id;

        await ctx.checkpoint(`epoch-${epoch}-complete`, {
          epoch,
          checkpoint_id: trained.checkpoint_id,
          mean_loss: trained.mean_loss,
          accuracy: trained.accuracy,
        });
      });

      const evaluated = (await ctx.step(
        'evaluate-network',
        async () => functions['evaluate-network']({ checkpoint_id: latestCheckpoint }),
        null,
        SETUP_STEP_OPTIONS,
      )) as EvaluateNetworkOutput;

      // The outcome is recorded as a checkpoint rather than a step. Every step must record a
      // portable operation, and "note that the target was met" is not one; a checkpoint is the
      // primitive meant for marking progress, and it shows up in the run history.
      await ctx.branch(ctx.when.gte(evaluated.accuracy, ctx.input.target_accuracy), {
        then: async () => {
          await ctx.checkpoint('target-met', {
            checkpoint_id: latestCheckpoint,
            accuracy: evaluated.accuracy,
          });
        },
        else: async () => {
          await ctx.checkpoint('target-missed', {
            checkpoint_id: latestCheckpoint,
            accuracy: evaluated.accuracy,
          });
        },
      });

      return {
        run_id: ctx.input.run_id,
        final_checkpoint: latestCheckpoint,
        epochs_completed: ctx.sampleInput.epochs.length,
      };
    },
  };
}
