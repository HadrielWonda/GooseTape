import DuctapeModule from '@ductape/sdk';
import { loadEnvironment, type GooseTapeEnvironment } from './environment.js';

// The SDK is published as CommonJS with an ESM wrapper, so the default export arrives either
// directly or nested depending on how the module is resolved.
const Ductape = (DuctapeModule as { default?: typeof DuctapeModule }).default ?? DuctapeModule;

/** A configured SDK client together with the environment it was built from. */
export interface DuctapeSession {
  readonly ductape: InstanceType<typeof Ductape>;
  readonly environment: GooseTapeEnvironment;
}

/**
 * Builds an SDK client from the validated environment.
 *
 * `DUCTAPE_FUNCTION_BASE_URL` is also exported into the process environment because the SDK
 * reads it directly when resolving portable function transports.
 *
 * @returns The client and the environment behind it.
 * @throws {Error} When required configuration is missing or malformed.
 */
export function openSession(): DuctapeSession {
  const environment = loadEnvironment();

  process.env['DUCTAPE_FUNCTION_BASE_URL'] = environment.functionBaseUrl;

  const ductape = new Ductape({
    accessKey: environment.accessKey,
    product: environment.product,
    env: environment.env,
  });

  return { ductape, environment };
}
