// A real-time check of the actual noVNC client, with no external npm packages.
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [browser, url] = process.argv.slice(2);
if (!browser || !url || typeof WebSocket !== 'function') {
  throw new Error('Usage: Node 22+ smoke_novnc_client.mjs CHROME LOCAL_URL');
}
const target = new URL(url);
if (target.hostname !== '127.0.0.1' || target.protocol !== 'http:') {
  throw new Error('The noVNC smoke test accepts only its local HTTP fixture');
}
const profile = await mkdtemp(join(tmpdir(), 'gsasii-browser-'));
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const child = spawn(browser, [
  '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
let browserError = '';
let launchError;
child.once('error', error => { launchError = error; });
child.stderr.on('data', data => { browserError = (browserError + data).slice(-4000); });
const pending = new Map();
const network = [];
let socket;
let sequence = 0;
let lastState = {};
const globalDeadline = Date.now() + 35000;
try {
  let port;
  const startupDeadline = Date.now() + 8000;
  while (Date.now() < startupDeadline) {
    if (launchError) throw launchError;
    if (child.exitCode !== null) throw new Error(`Chrome exited early: ${browserError}`);
    try { port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]; }
    catch { /* Chrome has not written its port yet. */ }
    if (port) break;
    await sleep(100);
  }
  if (!/^\d+$/.test(port || '')) throw new Error(`Chrome did not open DevTools: ${browserError}`);
  const response = await fetch(`http://127.0.0.1:${port}/json/list`, { signal: AbortSignal.timeout(3000) });
  if (!response.ok) throw new Error(`DevTools listing returned ${response.status}`);
  const page = (await response.json()).find(item => item.type === 'page');
  if (!page?.webSocketDebuggerUrl) throw new Error('Chrome did not create a page target');
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('DevTools WebSocket did not open')), 3000);
    socket.addEventListener('open', () => { clearTimeout(timer); resolve(); }, { once: true });
    socket.addEventListener('error', () => { clearTimeout(timer); reject(new Error('DevTools WebSocket failed')); }, { once: true });
  });
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const callback = pending.get(message.id);
      if (callback) {
        pending.delete(message.id);
        message.error ? callback.reject(new Error(JSON.stringify(message.error))) : callback.resolve(message.result);
      }
    } else if (message.method === 'Network.webSocketCreated') {
      network.push({ event: 'created', id: message.params.requestId, url: message.params.url });
    } else if (message.method === 'Network.webSocketHandshakeResponseReceived') {
      network.push({ event: 'handshake', id: message.params.requestId, status: message.params.response.status });
    } else if (message.method === 'Network.webSocketFrameError') {
      network.push({ event: 'error', id: message.params.requestId, error: message.params.errorMessage });
    }
  });
  const command = (method, params = {}) => {
    const id = ++sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`DevTools timed out: ${method}`)); }, 5000);
      pending.set(id, {
        resolve: value => { clearTimeout(timer); resolve(value); },
        reject: error => { clearTimeout(timer); reject(error); },
      });
      socket.send(JSON.stringify({ id, method, params }));
    });
  };
  await command('Network.enable');
  await command('Page.navigate', { url });
  // Virtual time can expire noVNC's timers before a real WebSocket handshake.
  // Poll in wall-clock time and require both the upgrade and the RFB UI state.
  const deadline = Math.min(globalDeadline, Date.now() + 25000);
  while (Date.now() < deadline) {
    const state = await command('Runtime.evaluate', {
      expression: `JSON.stringify({classes:document.documentElement.className,status:document.getElementById('noVNC_status')?.textContent||''})`,
      returnByValue: true,
    });
    if (state.result?.value) lastState = JSON.parse(state.result.value);
    if ((lastState.classes || '').split(/\s+/).includes('noVNC_connected')) {
      const created = network.find(item => item.event === 'created');
      if (!created || new URL(created.url).pathname.startsWith('//')) throw new Error('noVNC used a doubled-slash WebSocket path');
      if (!network.some(item => item.event === 'handshake' && item.id === created.id && item.status === 101)) {
        throw new Error('noVNC reported connected without a captured HTTP 101 handshake');
      }
      console.log(JSON.stringify({ connected: true, state: lastState, network }));
      process.exitCode = 0;
      break;
    }
    await sleep(250);
  }
  if (!(lastState.classes || '').split(/\s+/).includes('noVNC_connected')) {
    throw new Error(`The real noVNC client did not connect: ${JSON.stringify({ state: lastState, network })}`);
  }
} catch (error) {
  console.error(JSON.stringify({ failure: error.message, state: lastState, network, browserError }));
  throw error;
} finally {
  if (socket) {
    try { socket.close(); } catch { /* A failed opening may already be closed. */ }
  }
  child.kill('SIGTERM');
  await Promise.race([new Promise(resolve => child.once('exit', resolve)), sleep(1500)]);
  if (child.exitCode === null) child.kill('SIGKILL');
  await rm(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
}
