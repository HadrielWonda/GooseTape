import { defineFunctions } from '@ductape/sdk';
import type { PortableFunctionTransport } from '@ductape/sdk';

/**
 * The portable function contract the C# host implements.
 *
 * Ductape features never do matrix arithmetic themselves. They call these operations, which run
 * in the .NET host, and exchange small checkpoint identifiers rather than the weights
 * themselves — a 784x128x10 network is around a hundred thousand doubles, far too much to push
 * through a schema-validated function payload once per epoch.
 */
export const NEURAL_NETWORK_NAMESPACE = 'goosetape.neural-network';
export const NEURAL_NETWORK_VERSION = '1';

/** How long a single epoch may take before Ductape gives up on the invocation. */
const TRAIN_EPOCH_TIMEOUT_MS = 600_000;

/** Evaluation and classification are forward passes only, so they are far quicker. */
const INFERENCE_TIMEOUT_MS = 120_000;

const integer = { type: 'integer' } as const;
const number = { type: 'number' } as const;
const text = { type: 'string' } as const;

/**
 * Builds the signed HTTP transport pointing at the C# host.
 *
 * The path is the well-known route the Ductape SDK derives from a function base URL, so the
 * host and the SDK agree on it without either side inventing a convention.
 *
 * @param baseUrl - The host base URL, without a trailing slash.
 * @param operation - The operation the transport addresses.
 * @returns The transport declaration for that operation.
 */
function httpTransport(baseUrl: string, operation: string): PortableFunctionTransport {
  return {
    type: 'http',
    url:
      `${baseUrl}/.well-known/ductape/functions/` +
      `${encodeURIComponent(NEURAL_NETWORK_NAMESPACE)}/${encodeURIComponent(NEURAL_NETWORK_VERSION)}/${encodeURIComponent(operation)}`,
    authentication: 'ductape_hmac_sha256',
  };
}

/** Input to the `initialize-network` operation. */
export interface InitializeNetworkInput {
  readonly run_id: string;
  readonly topology: readonly number[];
  readonly seed: number;
}

/** Output of the `initialize-network` operation. */
export interface InitializeNetworkOutput {
  readonly checkpoint_id: string;
  readonly layer_count: number;
  readonly parameter_count: number;
  readonly input_size: number;
  readonly output_size: number;
}

/** Input to the `train-epoch` operation. */
export interface TrainEpochInput {
  readonly run_id: string;
  readonly epoch: number;
  readonly from_checkpoint: string;
  readonly learning_rate: number;
  readonly batch_size: number;
  readonly seed: number;
  readonly train_sample_size?: number;
}

/** Output of the `train-epoch` operation. */
export interface TrainEpochOutput {
  readonly checkpoint_id: string;
  readonly epoch: number;
  readonly mean_loss: number;
  readonly accuracy: number;
  readonly example_count: number;
  readonly duration_ms: number;
}

/** Input to the `evaluate-network` operation. */
export interface EvaluateNetworkInput {
  readonly checkpoint_id: string;
  readonly sample_size?: number;
}

/** Output of the `evaluate-network` operation. */
export interface EvaluateNetworkOutput {
  readonly checkpoint_id: string;
  readonly accuracy: number;
  readonly mean_loss: number;
  readonly example_count: number;
}

/** Input to the `classify-digit` operation. */
export interface ClassifyDigitInput {
  readonly checkpoint_id: string;
  readonly pixels: readonly number[];
}

/** Output of the `classify-digit` operation. */
export interface ClassifyDigitOutput {
  readonly digit: number;
  readonly confidence: number;
  readonly distribution: readonly number[];
}

/**
 * Builds the contract bound to a particular function host.
 *
 * @param baseUrl - The C# host base URL, without a trailing slash.
 * @returns The portable function contract.
 */
export function neuralNetworkContract(baseUrl: string) {
  return defineFunctions({
    namespace: NEURAL_NETWORK_NAMESPACE,
    version: NEURAL_NETWORK_VERSION,
    description: 'From-scratch neural network operations implemented in C# and invoked over signed HTTP.',
    operations: {
      'initialize-network': {
        description: 'Creates an untrained network and stores it as the epoch zero checkpoint.',
        timeout_ms: INFERENCE_TIMEOUT_MS,
        idempotent: true,
        transports: [httpTransport(baseUrl, 'initialize-network')],
        input: {
          type: 'object',
          properties: {
            run_id: text,
            topology: { type: 'array', items: integer, minItems: 3, maxItems: 8 },
            seed: integer,
          },
          required: ['run_id', 'topology', 'seed'],
        },
        output: {
          type: 'object',
          properties: {
            checkpoint_id: text,
            layer_count: integer,
            parameter_count: integer,
            input_size: integer,
            output_size: integer,
          },
          required: ['checkpoint_id', 'layer_count', 'parameter_count', 'input_size', 'output_size'],
        },
      },
      'train-epoch': {
        description: 'Runs one training epoch and stores the resulting network as a new checkpoint.',
        timeout_ms: TRAIN_EPOCH_TIMEOUT_MS,
        // Safe to retry: the starting checkpoint, epoch number and seed fully determine both the
        // shuffle order and every parameter update, so a retry recomputes an identical result.
        idempotent: true,
        transports: [httpTransport(baseUrl, 'train-epoch')],
        input: {
          type: 'object',
          properties: {
            run_id: text,
            epoch: integer,
            from_checkpoint: text,
            learning_rate: number,
            batch_size: integer,
            seed: integer,
            train_sample_size: integer,
          },
          required: ['run_id', 'epoch', 'from_checkpoint', 'learning_rate', 'batch_size', 'seed'],
        },
        output: {
          type: 'object',
          properties: {
            checkpoint_id: text,
            epoch: integer,
            mean_loss: number,
            accuracy: number,
            example_count: integer,
            duration_ms: number,
          },
          required: ['checkpoint_id', 'epoch', 'mean_loss', 'accuracy', 'example_count', 'duration_ms'],
        },
      },
      'evaluate-network': {
        description: 'Measures a checkpointed network against the held out test set.',
        timeout_ms: INFERENCE_TIMEOUT_MS,
        idempotent: true,
        transports: [httpTransport(baseUrl, 'evaluate-network')],
        input: {
          type: 'object',
          properties: { checkpoint_id: text, sample_size: integer },
          required: ['checkpoint_id'],
        },
        output: {
          type: 'object',
          properties: {
            checkpoint_id: text,
            accuracy: number,
            mean_loss: number,
            example_count: integer,
          },
          required: ['checkpoint_id', 'accuracy', 'mean_loss', 'example_count'],
        },
      },
      'classify-digit': {
        description: 'Classifies a single image with a checkpointed network.',
        timeout_ms: INFERENCE_TIMEOUT_MS,
        idempotent: true,
        transports: [httpTransport(baseUrl, 'classify-digit')],
        input: {
          type: 'object',
          properties: {
            checkpoint_id: text,
            pixels: { type: 'array', items: number, minItems: 1, maxItems: 65536 },
          },
          required: ['checkpoint_id', 'pixels'],
        },
        output: {
          type: 'object',
          properties: {
            digit: integer,
            confidence: number,
            distribution: { type: 'array', items: number },
          },
          required: ['digit', 'confidence', 'distribution'],
        },
      },
    },
  });
}

/** The contract type, for typing feature handlers against it. */
export type NeuralNetworkContract = ReturnType<typeof neuralNetworkContract>;
