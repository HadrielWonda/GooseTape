import { openSession } from '../config/ductape-client.js';

/**
 * Reports the status and per-step history of a feature run.
 *
 * This is the view that makes durability visible: it shows which epochs completed, which one
 * failed, and therefore where a resume would pick up from.
 *
 * Usage: `npm run status -- <feature-run-id>`
 */
async function main(): Promise<void> {
  const featureId = process.argv[2];

  if (featureId === undefined || featureId.trim().length === 0) {
    console.error('Usage: npm run status -- <feature-run-id>');
    process.exitCode = 1;
    return;
  }

  const { ductape, environment } = openSession();
  const scope = { product: environment.product, env: environment.env, feature_id: featureId };

  const status = await ductape.feature.status(scope);

  if (status === null) {
    console.error(`No run found with id ${featureId} in ${environment.product}/${environment.env}.`);
    process.exitCode = 1;
    return;
  }

  console.log(`Run ${featureId}`);
  console.log(`  status:       ${status.status}`);
  console.log(`  current step: ${status.current_step ?? 'none'}`);
  console.log('');

  const history = await ductape.feature.history(scope);

  console.log('Steps:');
  for (const step of history.steps ?? []) {
    console.log(`  ${step.status?.padEnd(10) ?? 'unknown   '} ${step.tag}`);
  }
}

await main();
