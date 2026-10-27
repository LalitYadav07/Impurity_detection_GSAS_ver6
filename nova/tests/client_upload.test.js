// Production expressions are exported from the built NOVA package, not copied here.
const assert = require('node:assert/strict');
const {File} = require('node:buffer');

async function tests(handlers) {
  const passed=[];
  for (const {handler, decode, start, kind} of handlers) {
    const create=new Function('trigger',`return (${handler})`);
    const invoke=(event,trigger)=>create(trigger)(event);
    for (const array of [false,true]) {
      const events=[];
      const bytes=Buffer.from('data_test\n_cell_length_a 4.1\n\t\u00b5');
      const file=new File([bytes],'test.cif');
      await invoke(array ? [file] : file,(name,args)=>events.push({name,args}));
      const reads=events.filter(e=>e.name===decode);
      assert.equal(reads.length,1);
      assert.equal(reads[0].args[1],'test.cif');
      assert.deepEqual(Buffer.from(reads[0].args[0]),bytes);
      if (start) {
        assert.equal(events.length,2);
        assert.equal(events[0].name,start);
        assert.deepEqual(events[0].args,['test.cif',kind]);
      } else assert.equal(events.length,1);
    }
    passed.push(`${decode}: File and File[] preserve name and exact bytes`);
    for (const empty of [null,undefined,[]]) {
      await invoke(empty,()=>assert.fail('Clearing an input transferred data'));
    }
    passed.push(`${decode}: clearing is safe`);
    const events=[];
    let release;
    const pending={name:'first.instprm',arrayBuffer:()=>new Promise(r=>{release=r;})};
    const first=invoke([pending],(name,args)=>events.push({name,args}));
    await invoke(new File(['second'],'second.yaml'),(name,args)=>events.push({name,args}));
    release(new TextEncoder().encode('first').buffer);
    await first;
    const reads=events.filter(e=>e.name===decode);
    assert.deepEqual(reads.map(e=>[e.args[1],Buffer.from(e.args[0]).toString()]),
      [['second.yaml','second'],['first.instprm','first']]);
    passed.push(`${decode}: asynchronous reads retain their own selected file`);
  }
  return {passed:passed.length,checks:passed};
}

module.exports=tests;
if (require.main===module) {
  const handlers=JSON.parse(require('node:fs').readFileSync(process.argv[2],'utf8'));
  tests(handlers).then(result=>console.log(JSON.stringify(result))).catch(error=>{
    console.error(error);process.exitCode=1;
  });
}
