import { inspectFeatureHandlerControlFlow } from '@ductape/sdk/dist/features/feature-control-flow.validator';
import {
  trainDigitRecogniserFeature,
  DEFAULT_TRAINING_RUN,
} from '../features/train-digit-recogniser.feature.js';
import { classifyDigitFeature } from '../features/classify-digit.feature.js';

/**
 * Checks every feature handler against Ductape's portable-mode control flow rules.
 *
 * Ductape compiles a feature by recording its handler once, so a native `if`, ternary, `for` or
 * `while` would run at record time rather than at feature runtime and silently bake the wrong
 * step graph. The SDK runs this same validator over `handler.toString()` when a feature is
 * defined; running it here catches a violation locally, before anything is published, and
 * without needing Ductape credentials.
 */
const PLACEHOLDER_PRODUCT = 'compile-check';
const PLACEHOLDER_BASE_URL = 'http://127.0.0.1:5207';

function main(): void {
  const definitions = [
    trainDigitRecogniserFeature(PLACEHOLDER_PRODUCT, PLACEHOLDER_BASE_URL, DEFAULT_TRAINING_RUN),
    classifyDigitFeature(PLACEHOLDER_PRODUCT, PLACEHOLDER_BASE_URL, {
      checkpoint_id: 'sample-epoch-0',
      pixels: Array.from({ length: 20 }, () => 0),
    }),
  ];

  let failures = 0;

  for (const definition of definitions) {
    const mode = definition.controlFlowMode ?? 'portable';
    const diagnostics = inspectFeatureHandlerControlFlow(definition.handler.toString(), mode);

    if (diagnostics.length === 0) {
      console.log(`  ok   ${definition.tag} (${mode})`);
      continue;
    }

    failures += diagnostics.length;
    console.error(`  FAIL ${definition.tag} (${mode})`);

    for (const diagnostic of diagnostics) {
      console.error(
        `       ${diagnostic.line}:${diagnostic.column} ${diagnostic.code} ${diagnostic.message}`,
      );
    }
  }

  if (failures > 0) {
    console.error(`\n${failures} portable-mode violation(s) found.`);
    process.exitCode = 1;
    return;
  }

  console.log('\nAll feature handlers are portable-mode clean.');
}

main();
