import DuctapeModule from '@ductape/sdk';
import type DuctapeClient from '@ductape/sdk/dist/index';
import { loadEnvironment, type GooseTapeEnvironment } from './environment.js';

/**
 * The SDK client constructor, typed from the CommonJS declarations.
 *
 * `@ductape/sdk@0.3.7` ships ESM typings that declare the default export as
 * `typeof sdk.default`, where `sdk` is already the class. That type does not resolve, and with
 * `skipLibCheck` on it silently becomes `any`, disabling type checking on every client call.
 * Typing from `dist/index`, whose declarations are correct, restores it.
 */
type DuctapeConstructor = new (init: ConstructorParameters<typeof DuctapeClient>[0]) => DuctapeClient;

// The package is CommonJS with an ESM wrapper, so the class arrives either directly or nested
// under `default` depending on how the module is resolved.
const Ductape: DuctapeConstructor =
  (DuctapeModule as unknown as { default?: DuctapeConstructor }).default ??
  (DuctapeModule as unknown as DuctapeConstructor);

/** A configured SDK client together with the environment it was built from. */
export interface DuctapeSession {
  readonly ductape: DuctapeClient;
  readonly environment: GooseTapeEnvironment;
}

/** Options for {@link openSession}. */
export interface SessionOptions {
  /**
   * Whether the client keeps a runtime snapshot of the configured product in sync.
   *
   * Defaults to true. Turn it off only when the configured product may not exist yet: with sync
   * on, the SDK bootstraps that product while authenticating, so every call — including the one
   * that would create the product — fails with "Product/Integration not found".
   */
  readonly runtimeSync?: boolean;
}

/**
 * Builds an SDK client from the validated environment.
 *
 * `DUCTAPE_FUNCTION_BASE_URL` is also exported into the process environment because the SDK
 * reads it directly when resolving portable function transports.
 *
 * @param options - Client options.
 * @returns The client and the environment behind it.
 * @throws {Error} When required configuration is missing or malformed.
 */
export function openSession(options: SessionOptions = {}): DuctapeSession {
  const environment = loadEnvironment();

  process.env['DUCTAPE_FUNCTION_BASE_URL'] = environment.functionBaseUrl;

  const ductape = new Ductape({
    accessKey: environment.accessKey,
    product: environment.product,
    env: environment.env,
    runtime_sync: options.runtimeSync ?? true,
  });

  return { ductape, environment };
}

/** How long {@link flushPendingWrites} waits for Ductape to accept buffered records. */
const FLUSH_DEADLINE_MS = 30_000;

/** How often a flush re-checks the queue while waiting on a retry. */
const FLUSH_POLL_MS = 250;

/**
 * Waits until the SDK has delivered every buffered run record, or the deadline passes.
 *
 * The SDK sends feature and step results through an internal write queue. `close()` does not
 * drain it, so `await ductape.close()` followed by `process.exit()` drops whatever is still
 * queued: measured three out of three times, the final execution record was never sent. On a
 * plain exit delivery happens anyway, because the in-flight request keeps Node alive, but that
 * relies on an implementation detail. Flushing here makes delivery deterministic. The queue is
 * not exported from the package root, so it is reached by its dist path.
 * Reported upstream as Ductape-LLC/ductape-emails#28.
 *
 * @returns The number of records still undelivered when the deadline passed; zero on success.
 */
export async function flushPendingWrites(): Promise<number> {
  const { resilientWriteQueue } = await import('@ductape/sdk/dist/runtime/resilient-write-queue');
  const deadline = Date.now() + FLUSH_DEADLINE_MS;
  const buffered = resilientWriteQueue.size();

  if (buffered > 0) {
    console.log(`Delivering ${buffered} buffered run record batch(es) to Ductape...`);
  }

  while (resilientWriteQueue.size() > 0 && Date.now() < deadline) {
    await resilientWriteQueue.flushDue();
    await new Promise((resolve) => setTimeout(resolve, FLUSH_POLL_MS));
  }

  return resilientWriteQueue.size();
}
