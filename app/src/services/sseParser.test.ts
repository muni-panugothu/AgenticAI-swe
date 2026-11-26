import assert from 'node:assert/strict';
import { consumeSseBuffer, parseSseDataPayload } from './sseParser';

function run(name: string, fn: () => void) {
  try {
    fn();
    console.log(`ok - ${name}`);
  } catch (err) {
    console.error(`fail - ${name}`);
    throw err;
  }
}

run('parses token/meta/done', () => {
  assert.deepEqual(parseSseDataPayload('{"type":"token","token":"Hi"}'), {
    type: 'token',
    token: 'Hi',
  });
  assert.equal(parseSseDataPayload('{"type":"meta","sessionId":"s1"}')?.type, 'meta');
  assert.equal(parseSseDataPayload('{"type":"done"}')?.type, 'done');
});

run('maps CANCELLED error to cancelled', () => {
  const evt = parseSseDataPayload(
    '{"type":"error","code":"CANCELLED","error":"The request was cancelled."}'
  );
  assert.equal(evt?.type, 'cancelled');
});

run('multiple events in one chunk', () => {
  const events: unknown[] = [];
  consumeSseBuffer(
    'data: {"type":"token","token":"A"}\n\ndata: {"type":"token","token":"B"}\n\n',
    (e) => events.push(e)
  );
  assert.equal(events.length, 2);
});

run('split across chunks', () => {
  const events: unknown[] = [];
  let buf = consumeSseBuffer('data: {"type":"token","token":"hel', (e) =>
    events.push(e)
  );
  buf = consumeSseBuffer(buf + 'lo"}\n\n', (e) => events.push(e));
  assert.equal(events.length, 1);
  assert.deepEqual(events[0], { type: 'token', token: 'hello' });
  assert.equal(buf, '');
});

run('ignores comments and keeps heartbeat', () => {
  const events: unknown[] = [];
  consumeSseBuffer(
    ': ping\n\ndata: {"type":"heartbeat","ts":1}\n\n',
    (e) => events.push(e)
  );
  assert.equal(events.length, 1);
  assert.equal((events[0] as { type: string }).type, 'heartbeat');
});

run('unknown events ignored', () => {
  const events: unknown[] = [];
  consumeSseBuffer(
    'data: {"type":"future_feature","x":1}\n\n',
    (e) => events.push(e)
  );
  assert.equal(events.length, 0);
});

run('flush without trailing newline', () => {
  const events: unknown[] = [];
  let buf = consumeSseBuffer(
    'data: {"type":"token","token":"Z"}',
    (e) => events.push(e)
  );
  assert.equal(events.length, 0);
  consumeSseBuffer(`${buf}\n\n`, (e) => events.push(e));
  assert.deepEqual(events[0], { type: 'token', token: 'Z' });
});

console.log('All sseParser tests passed');
