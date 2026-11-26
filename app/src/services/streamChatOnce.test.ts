/**
 * Batch A — Expo single-POST streaming guarantees.
 * Run from app/: npx tsx src/services/streamChatOnce.test.ts
 */
import assert from 'node:assert/strict';
import {
  selectStreamTransport,
  streamChatOnce,
  StreamCancelledError,
  StreamDeliveryError,
  StreamTransportUnavailableError,
  type StreamTransportCapability,
} from './streamChatOnce';

let passed = 0;
let failed = 0;

async function run(name: string, fn: () => void | Promise<void>) {
  try {
    await fn();
    passed += 1;
    console.log(`ok - ${name}`);
  } catch (err) {
    failed += 1;
    console.error(`fail - ${name}`);
    console.error(err);
  }
}

function caps(partial: Partial<StreamTransportCapability>): StreamTransportCapability {
  return {
    hasReadableStream: false,
    hasXmlHttpRequest: false,
    expoFetch: null,
    ...partial,
  };
}

async function main() {
  await run('selectStreamTransport prefers expo-fetch when capable', () => {
    const kind = selectStreamTransport(
      caps({
        hasReadableStream: true,
        expoFetch: (async () => new Response()) as typeof fetch,
      })
    );
    assert.equal(kind, 'expo-fetch');
  });

  await run('selectStreamTransport falls back to xhr before any POST', () => {
    const kind = selectStreamTransport(
      caps({ hasXmlHttpRequest: true, expoFetch: null })
    );
    assert.equal(kind, 'xhr');
  });

  await run('selectStreamTransport throws when no transport available', () => {
    assert.throws(
      () => selectStreamTransport(caps({})),
      (e: unknown) => e instanceof StreamTransportUnavailableError
    );
  });

  await run(
    'expo-fetch 200 without reader does not POST via XHR or JSON',
    async () => {
      let expoPosts = 0;
      let xhrPosts = 0;
      let jsonPosts = 0;

      await assert.rejects(
        () =>
          streamChatOnce({
            url: 'http://test/api/chat/stream',
            body: JSON.stringify({ message: 'hi' }),
            headers: { Authorization: 'Bearer x' },
            handlers: {
              onToken: () => undefined,
            },
            deps: {
              detectCapabilities: async () =>
                caps({
                  hasReadableStream: true,
                  hasXmlHttpRequest: true,
                  expoFetch: (async () => new Response()) as typeof fetch,
                }),
            },
            executors: {
              runExpoFetch: async () => {
                expoPosts += 1;
                throw new StreamDeliveryError(
                  'Streaming response was accepted but this runtime cannot read the body progressively. ' +
                    'The request was not retried to avoid duplicate agent execution.',
                  { requestSubmitted: true }
                );
              },
              runXhr: async () => {
                xhrPosts += 1;
              },
            },
          }),
        (e: unknown) => e instanceof StreamDeliveryError
      );

      assert.equal(jsonPosts, 0);
      assert.equal(xhrPosts, 0);
      assert.equal(expoPosts, 1);
    }
  );

  await run(
    'tool-heavy stream with zero tokens does not authorize a second POST',
    async () => {
      let posts = 0;
      const result = await streamChatOnce({
        url: 'http://test/api/chat/stream',
        body: JSON.stringify({ message: 'mutate memory' }),
        headers: {},
        handlers: { onToken: () => undefined },
        deps: {
          forceTransport: 'xhr',
          detectCapabilities: async () => caps({ hasXmlHttpRequest: true }),
        },
        executors: {
          runXhr: async ({ onEvent }) => {
            posts += 1;
            onEvent({
              type: 'meta',
              sessionId: 's1',
              action: 'general',
              contextUsed: false,
            });
            onEvent({ type: 'done', sessionId: 's1', model: 'm', provider: 'p' });
          },
          runExpoFetch: async () => {
            posts += 1;
          },
        },
      });
      assert.equal(posts, 1);
      assert.equal(result.fullText, '');
      assert.equal(result.requestSubmitted, true);
    }
  );

  await run(
    'stream fails after partial tokens — preserves partial, no second POST',
    async () => {
      let posts = 0;
      let lastPartial = '';
      await assert.rejects(
        () =>
          streamChatOnce({
            url: 'http://test/api/chat/stream',
            body: '{}',
            headers: {},
            handlers: {
              onToken: (_t, full) => {
                lastPartial = full;
              },
            },
            deps: {
              forceTransport: 'xhr',
              detectCapabilities: async () => caps({ hasXmlHttpRequest: true }),
            },
            executors: {
              runXhr: async ({ onEvent }) => {
                posts += 1;
                onEvent({ type: 'token', token: 'Hel' });
                onEvent({ type: 'token', token: 'lo' });
                throw new StreamDeliveryError('connection reset', {
                  requestSubmitted: true,
                });
              },
            },
          }),
        (e: unknown) => {
          assert.ok(e instanceof StreamDeliveryError);
          assert.equal(e.partialText, 'Hello');
          assert.equal(e.requestSubmitted, true);
          return true;
        }
      );
      assert.equal(posts, 1);
      assert.equal(lastPartial, 'Hello');
    }
  );

  await run(
    'stream fails before any visible tokens — no second POST',
    async () => {
      let posts = 0;
      await assert.rejects(
        () =>
          streamChatOnce({
            url: 'http://test/api/chat/stream',
            body: '{}',
            headers: {},
            handlers: { onToken: () => undefined },
            deps: {
              forceTransport: 'expo-fetch',
              detectCapabilities: async () =>
                caps({
                  hasReadableStream: true,
                  expoFetch: (async () => new Response()) as typeof fetch,
                }),
            },
            executors: {
              runExpoFetch: async () => {
                posts += 1;
                throw new StreamDeliveryError('mid-flight failure', {
                  requestSubmitted: true,
                  partialText: '',
                });
              },
              runXhr: async () => {
                posts += 1;
              },
            },
          }),
        (e: unknown) => e instanceof StreamDeliveryError
      );
      assert.equal(posts, 1);
    }
  );

  await run('user cancel — StreamCancelledError, no fallback POST', async () => {
    let posts = 0;
    const controller = new AbortController();
    await assert.rejects(
      () =>
        streamChatOnce({
          url: 'http://test/api/chat/stream',
          body: '{}',
          headers: {},
          handlers: {
            signal: controller.signal,
            onToken: () => undefined,
          },
          deps: {
            forceTransport: 'xhr',
            detectCapabilities: async () => caps({ hasXmlHttpRequest: true }),
          },
          executors: {
            runXhr: async () => {
              posts += 1;
              controller.abort();
              throw new StreamCancelledError();
            },
            runExpoFetch: async () => {
              posts += 1;
            },
          },
        }),
      (e: unknown) => e instanceof StreamCancelledError
    );
    assert.equal(posts, 1);
  });

  await run(
    'unavailable transport before send — clear error, zero POSTs',
    async () => {
      let posts = 0;
      await assert.rejects(
        () =>
          streamChatOnce({
            url: 'http://test/api/chat/stream',
            body: '{}',
            headers: {},
            handlers: { onToken: () => undefined },
            deps: {
              detectCapabilities: async () => caps({}),
            },
            executors: {
              runExpoFetch: async () => {
                posts += 1;
              },
              runXhr: async () => {
                posts += 1;
              },
            },
          }),
        (e: unknown) => e instanceof StreamTransportUnavailableError
      );
      assert.equal(posts, 0);
    }
  );

  await run(
    'forced xhr transport is used once when expo-fetch cannot stream',
    async () => {
      let xhrPosts = 0;
      let expoPosts = 0;
      await streamChatOnce({
        url: 'http://test/api/chat/stream',
        body: '{}',
        headers: {},
        handlers: { onToken: () => undefined },
        deps: {
          detectCapabilities: async () =>
            caps({
              hasXmlHttpRequest: true,
              hasReadableStream: false,
              expoFetch: (async () => {
                expoPosts += 1;
                return new Response('should not run');
              }) as typeof fetch,
            }),
        },
        executors: {
          runXhr: async ({ onEvent }) => {
            xhrPosts += 1;
            onEvent({ type: 'token', token: 'ok' });
            onEvent({ type: 'done' });
          },
          runExpoFetch: async () => {
            expoPosts += 1;
          },
        },
      });
      assert.equal(xhrPosts, 1);
      assert.equal(expoPosts, 0);
    }
  );

  await run(
    'rapid duplicate send attempts — in-flight gate allows one POST',
    async () => {
      let posts = 0;
      const inFlight = { current: false };

      const gatedSend = async () => {
        if (inFlight.current) return;
        inFlight.current = true;
        try {
          await streamChatOnce({
            url: 'http://test/api/chat/stream',
            body: JSON.stringify({ message: 'same' }),
            headers: {},
            handlers: { onToken: () => undefined },
            deps: {
              forceTransport: 'xhr',
              detectCapabilities: async () => caps({ hasXmlHttpRequest: true }),
            },
            executors: {
              runXhr: async ({ onEvent }) => {
                posts += 1;
                await new Promise((r) => setTimeout(r, 20));
                onEvent({ type: 'done' });
              },
            },
          });
        } finally {
          inFlight.current = false;
        }
      };

      await Promise.all([gatedSend(), gatedSend(), gatedSend()]);
      assert.equal(posts, 1);
    }
  );

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}

main();
