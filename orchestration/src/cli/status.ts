import { openSession } from '../config/ductape-client.js';

/**
 * Reports the status and checkpoints of a feature run.
 *
 * Limitation on `@ductape/sdk` 0.3.7: `feature.status` reads Ductape's workflow store, and runs
 * started with `feature.execute` are recorded in the processor store instead, so this finds
 * nothing for them. It is kept for runs that do reach the workflow store. See the README's
 * Durability section.
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
    console.error('Runs started with feature.execute are not visible to feature.status on @ductape/sdk 0.3.7.');
    console.error('See the Durability section of the README.');
    process.exitCode = 1;
    return;
  }

  console.log(`Run ${featureId}`);
  console.log(`  status:          ${status.status}`);
  console.log(`  current step:    ${status.current_step ?? 'none'}`);
  console.log(`  completed steps: ${status.completed_steps.join(', ') || 'none'}`);
  console.log('');

  // Checkpoints are what a resume picks up from, so they are the useful view of progress.
  const history = await ductape.feature.history(scope);

  console.log('Checkpoints:');
  for (const checkpoint of history.checkpoints) {
    const at = new Date(checkpoint.timestamp).toISOString();
    console.log(`  ${at}  ${checkpoint.name}  ${JSON.stringify(checkpoint.metadata ?? {})}`);
  }
}

await main();
