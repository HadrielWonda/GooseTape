import 'dotenv/config';

/**
 * The configuration this orchestration layer needs, validated once at startup.
 *
 * Secrets are read from the environment and never committed. `.env` is git-ignored; copy
 * `.env.example` and fill it in, or export the variables another way.
 */
export interface GooseTapeEnvironment {
  /** Ductape SDK access key. Also the shared secret the C# function host verifies with. */
  readonly accessKey: string;
  /** Ductape workspace identifier. */
  readonly workspaceId: string;
  /** Ductape publishable key, safe to expose to clients. */
  readonly publishableKey: string;
  /** Product tag features are published under. */
  readonly product: string;
  /** Environment slug features run in, such as `dev` or `prd`. */
  readonly env: string;
  /** Base URL of the C# portable function host, without a trailing slash. */
  readonly functionBaseUrl: string;
}

const REQUIRED_VARIABLES = [
  'DUCTAPE_ACCESS_KEY',
  'DUCTAPE_WORKSPACE',
  'DUCTAPE_PUBLISHABLE_KEY',
  'DUCTAPE_PRODUCT',
  'DUCTAPE_ENV',
  'DUCTAPE_FUNCTION_BASE_URL',
] as const;

type RequiredVariable = (typeof REQUIRED_VARIABLES)[number];

/**
 * Reads and validates the environment.
 *
 * Fails immediately and names every missing variable at once, rather than failing on the first
 * one and making the caller rediscover the rest one run at a time.
 *
 * @returns The validated environment.
 * @throws {Error} When a required variable is missing, or the function base URL is malformed.
 */
export function loadEnvironment(): GooseTapeEnvironment {
  const missing = REQUIRED_VARIABLES.filter((name) => isBlank(process.env[name]));

  if (missing.length > 0) {
    throw new Error(
      `Missing required environment variables: ${missing.join(', ')}. ` +
        'Copy orchestration/.env.example to orchestration/.env and fill it in.',
    );
  }

  const functionBaseUrl = stripTrailingSlash(read('DUCTAPE_FUNCTION_BASE_URL'));
  assertSecureFunctionBaseUrl(functionBaseUrl);

  return {
    accessKey: read('DUCTAPE_ACCESS_KEY'),
    workspaceId: read('DUCTAPE_WORKSPACE'),
    publishableKey: read('DUCTAPE_PUBLISHABLE_KEY'),
    product: read('DUCTAPE_PRODUCT'),
    env: read('DUCTAPE_ENV'),
    functionBaseUrl,
  };
}

/**
 * Mirrors the transport rule the Ductape SDK enforces, so a misconfigured URL fails here with a
 * clear message rather than deep inside an invocation.
 *
 * @param value - The configured base URL.
 * @throws {Error} When the URL is unparseable, or is plain HTTP against a non-local host.
 */
function assertSecureFunctionBaseUrl(value: string): void {
  let parsed: URL;

  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`DUCTAPE_FUNCTION_BASE_URL is not a valid URL: ${JSON.stringify(value)}.`);
  }

  const isLocal =
    parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1' || parsed.hostname === '::1';

  if (parsed.protocol !== 'https:' && !(isLocal && parsed.protocol === 'http:')) {
    throw new Error(
      'DUCTAPE_FUNCTION_BASE_URL must use HTTPS. Plain HTTP is only accepted for localhost development.',
    );
  }
}

function read(name: RequiredVariable): string {
  return process.env[name] as string;
}

function isBlank(value: string | undefined): boolean {
  return value === undefined || value.trim().length === 0;
}

function stripTrailingSlash(value: string): string {
  return value.replace(/\/+$/, '');
}
