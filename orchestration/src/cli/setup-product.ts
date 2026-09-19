import { openSession } from '../config/ductape-client.js';

/**
 * Creates the Ductape product and environment this project publishes into, if they are missing.
 *
 * Idempotent: an existing product or environment is left untouched, so running it twice is
 * harmless. Ductape derives the product tag from the name server-side, and the tag it settles on
 * is printed so `DUCTAPE_PRODUCT` can be checked against it.
 *
 * Usage: `npm run setup`
 */
const PRODUCT_DESCRIPTION =
  'GooseTape: a from-scratch C# neural network trained as a durable Ductape feature.';

/** Ductape environment slugs are exactly three characters. */
const ENVIRONMENT_NAMES: Readonly<Record<string, string>> = {
  dev: 'Development',
  stg: 'Staging',
  prd: 'Production',
};

/** The product fields this command reads. The SDK does not export its product types. */
interface CreatedProduct {
  readonly tag?: string;
  readonly name?: string;
}

async function main(): Promise<void> {
  // Runtime sync would try to bootstrap the very product this command is about to create.
  const { ductape, environment } = openSession({ runtimeSync: false });

  const existing = await tryFetchProduct(() => ductape.product.fetch(environment.product));
  let productTag = environment.product;

  if (existing === null) {
    const name = productNameFrom(environment.product);
    console.log(`Creating product "${name}"...`);
    const created = (await ductape.product.create({
      name,
      description: PRODUCT_DESCRIPTION,
    } as unknown as Parameters<typeof ductape.product.create>[0])) as CreatedProduct;

    console.log(`  created: name="${created.name ?? name}" tag="${created.tag ?? '(not returned)'}"`);
    warnOnTagMismatch(created.tag, environment.product);
    productTag = created.tag ?? productTag;
  } else {
    console.log(`Product "${environment.product}" already exists; leaving it as is.`);
  }

  await ensureEnvironment(ductape, productTag, environment.env);
  console.log('\nSetup complete. Run `npm run doctor` to verify.');
}

async function ensureEnvironment(
  ductape: ReturnType<typeof openSession>['ductape'],
  product: string,
  slug: string,
): Promise<void> {
  const environments = (await ductape.product.environments.list(product)) as ReadonlyArray<{ slug: string }>;

  if (environments.some((entry) => entry.slug === slug)) {
    console.log(`Environment "${slug}" already exists; leaving it as is.`);
    return;
  }

  console.log(`Creating environment "${slug}"...`);
  await ductape.product.environments.create(product, {
    env_name: ENVIRONMENT_NAMES[slug] ?? slug,
    description: `GooseTape ${ENVIRONMENT_NAMES[slug] ?? slug} environment.`,
    slug,
    active: true,
  });
  console.log('  created.');
}

/**
 * Ductape tags products as `<workspace>:<name>`, so a configured tag in that form names the
 * product by its final segment. Creating under the full tag would produce a second product.
 */
function productNameFrom(tag: string): string {
  return tag.split(':').at(-1) ?? tag;
}

async function tryFetchProduct(fetch: () => Promise<unknown>): Promise<unknown> {
  try {
    return await fetch();
  } catch {
    return null;
  }
}

function warnOnTagMismatch(actualTag: string | undefined, configuredTag: string): void {
  if (actualTag === undefined || actualTag === configuredTag) {
    return;
  }

  console.warn(
    `\n  NOTE: Ductape assigned the tag "${actualTag}", not "${configuredTag}". ` +
      `Set DUCTAPE_PRODUCT=${actualTag} in orchestration/.env.`,
  );
}

await main();
