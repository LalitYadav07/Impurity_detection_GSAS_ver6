// Run with Node, or the desktop's bundled Node test runtime.
const assert = require("node:assert/strict");
const plugin = require("../src/radar_pd_nova/web/transition_cursor.js");
const {CursorState, LatestRenderer} = plugin;
const server = (cursor = 5, scope = "one", runIds = ["a","b","c","d","e","f"]) =>
  ({cursor, scope, runIds, epoch: "epoch", frames: [], figure: {data: [], layout: {}}});
const ack = (model, revision, accepted = true) => ({client_id: model.clientId, revision, accepted});

async function tests() {
  const passed = [];
  {
    const m = new CursorState("tab");
    m.begin(server(), true); m.input(0); const first = m.release();
    m.begin(server(), true); m.input(4);
    assert.equal(m.reconcile(server(0), ack(m, first.revision)), false);
    assert.equal(m.index, 4); // an old reply must not pull the thumb back to zero
    const second = m.release();
    assert.equal(m.reconcile(server(0), ack(m, first.revision)), false);
    assert.equal(m.index, 4);
    assert.equal(m.reconcile(server(4), ack(m, second.revision)), true);
    assert.equal(m.index, 4);
    assert.equal(m.owns("one"), false);
    passed.push("older acknowledgements cannot move a newer drag or pending selection");
  }
  {
    const m = new CursorState("tab"); m.begin(server(), true);
    for (const i of [4,3,2,1,0,2,4,3]) {
      m.input(i); m.reconcile(server(5), ack(m, 0)); assert.equal(m.index, i);
    }
    const last = m.release();
    assert.equal(m.release(), null); // change / pointerup / blur / lostcapture
    assert.equal(m.release(), null);
    assert.equal(m.reconcile(server(5), ack(m, last.revision)), false);
    assert.equal(m.index, 3); // wait for the selected run and acknowledgement together
    assert.equal(m.reconcile(server(3), ack(m, last.revision)), true);
    passed.push("rapid reversals remain local and one release is submitted per gesture");
  }
  {
    const m = new CursorState("keyboard");
    m.begin(server(), false); m.input(1); const old = m.release();
    m.begin(server(), false); m.input(2); const latest = m.release();
    assert.equal(m.reconcile(server(1), ack(m, old.revision)), false);
    assert.equal(m.index, 2);
    assert.equal(m.reconcile(server(2), ack(m, latest.revision)), true);
    passed.push("keyboard selections also reject late replies");
  }
  {
    const m = new CursorState("tab");
    m.begin(server(), true); m.input(2);
    m.reconcile(server(6, "one", ["new","a","b","c","d","e","f"]), {});
    assert.equal(m.bundle.runIds[m.index], "c");
    const last = m.release(); assert.equal(last.run_id, "c");
    m.reconcile(server(3, "one", ["new","a","b","c","d","e","f"]), ack(m, last.revision));
    assert.equal(m.index, 3);
    passed.push("arrivals preserve stable run identity until release");
  }
  {
    const m = new CursorState("new-tab");
    m.begin(server(), true); m.input(0); const last = m.release();
    assert.equal(m.reconcile(server(5), {client_id: "old-tab", revision: 100, accepted: true}), false);
    assert.equal(m.index, 0);
    assert.equal(m.reconcile(server(5), ack(m, last.revision, false)), true);
    m.begin(server(), true); m.input(3);
    assert.equal(m.reconcile(server(1, "new-experiment"), {}), true);
    assert.equal(m.index, 1); assert.equal(m.dragging, false); assert.equal(m.pending, false);
    passed.push("new tabs, rejected requests and experiment changes cancel safely");
  }
  {
    let active = 0, maximum = 0; const drawn = [], completed = [], gates = [];
    const renderer = new LatestRenderer(async (packet) => {
      active++; maximum = Math.max(maximum, active); drawn.push(packet);
      await new Promise(resolve => gates.push(resolve)); active--;
    }, packet => completed.push(packet), error => { throw error; });
    renderer.enqueue(4); renderer.enqueue(0); renderer.enqueue(5); renderer.enqueue(1);
    assert.deepEqual(drawn, [4]);
    gates.shift()(); await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
    assert.deepEqual(drawn, [4,1]); // stale zero and five are discarded
    renderer.enqueue(2); renderer.enqueue(3);
    gates.shift()(); await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
    assert.deepEqual(drawn, [4,1,3]);
    gates.shift()(); await renderer.done;
    assert.equal(maximum, 1); assert.deepEqual(completed, [4,1,3]);
    passed.push("one asynchronous Plotly draw at a time; only the latest queued frame survives");
  }
  {
    let release; const completed = [];
    const renderer = new LatestRenderer(() => new Promise(r => {release = r;}),
      p => completed.push(p), e => {throw e;});
    renderer.enqueue(1); renderer.enqueue(2); renderer.stop(); release(); await renderer.done;
    assert.deepEqual(completed, []);
    passed.push("unmount drops pending drawings");
  }
  {
    let component;
    plugin.install({component: (_name, definition) => {component = definition;}});
    let current = server();
    current.figure = {data:[{type:"heatmap"},{type:"scatter"}],layout:{yaxis2:{}}};
    current.frames = current.runIds.map(id => ({trace:{name:id},annotations:[],shapes:[],axis:{}}));
    const m = new CursorState("tab"); m.begin(current,true); m.input(3);
    const packets=[];
    const context={model:m,scope:"one",cursor:5,_renderer:{enqueue:p=>packets.push(p)},
      $refs:{plot:{}},snapshot:()=>current};
    component.methods.enqueue.call(context,false);
    const last=m.release();
    current={...current,cursor:3};context.cursor=3;m.reconcile(current,ack(m,last.revision));
    component.methods.enqueue.call(context,false);
    assert.equal(packets.length,1); // the acknowledgement must not redraw the same scan
    current={...current,epoch:"new-zoom-or-display"};
    component.methods.enqueue.call(context,false);
    assert.equal(packets.length,2);
    component.methods.enqueue.call(context,true); // resize still redraws
    assert.equal(packets.length,3);
    passed.push("acknowledging a displayed scan does not draw it twice; display and resize changes still refresh");
  }
  return {passed: passed.length, checks: passed};
}
if (require.main === module) tests().then(result => console.log(JSON.stringify(result, null, 2)));
module.exports = tests;
