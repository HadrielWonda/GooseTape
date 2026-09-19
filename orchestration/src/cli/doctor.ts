import { openSession } from '../config/ductape-client.js';

/** The only product fields this check reads. The SDK does not export its product types. */
interface ProductSummary {
  readonly name?: string;
}

/** The only environment field this check reads. */
interface EnvironmentSummary {
  readonly slug: string;
}

/**
 * Checks, in order, everything a training run depends on, and stops at the first failure with
 * a message naming what to fix.
 *
 * 1. The access key authenticates against the workspace.
 * 2. The configured product exists.
 * 3. The configured environment exists in that product.
 * 4. The C# function host is reachable and serves the expected contract.
 *
 * Read-only: it creates, publishes and executes nothing.
 *
 * Usage: `npm run doctor`
 */
async function main(): Promise<void> {
  const { ductape, environment } = openSession();

  console.log(`Workspace ${environment.workspaceId}`);

  const product = (await check(`product "${environment.product}" exists`, () =>
    ductape.product.fetch(environment.product),
  )) as ProductSummary;
  console.log(`       name: ${product.name ?? '(unnamed)'}`);

  const environments = (await check(`environments of "${environment.product}" are readable`, () =>
    ductape.product.environments.list(environment.product),
  )) as readonly EnvironmentSummary[];
  const slugs = environments.map((entry) => entry.slug);
  console.log(`       slugs: ${slugs.join(', ') || '(none)'}`);

  if (!slugs.includes(environment.env)) {
    fail(`environment "${environment.env}" is not one of: ${slugs.join(', ') || '(none)'}. Set DUCTAPE_ENV to one of them.`);
    return;
  }
  console.log(`  ok   environment "${environment.env}" exists`);

  const health = await check(`function host at ${environment.functionBaseUrl} is up`, async () => {
    const response = await fetch(`${environment.functionBaseUrl}/health`);
    return (await response.json()) as { function_namespace: string; operations: string[] };
  });
  console.log(`       serves ${health.function_namespace}: ${health.operations.join(', ')}`);

  console.log('\nReady to publish and train.');
}

async function check<T>(description: string, probe: () => Promise<T>): Promise<T> {
  try {
    const result = await probe();
    console.log(`  ok   ${description}`);
    return result;
  } catch (error) {
    fail(`${description}: ${error instanceof Error ? error.message : String(error)}`);
    throw error;
  }
}

function fail(message: string): void {
  console.error(`  FAIL ${message}`);
  process.exitCode = 1;
}

await main().catch(() => {
  process.exitCode = 1;
});
