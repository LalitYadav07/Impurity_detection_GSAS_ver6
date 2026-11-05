/* Run the real emitted handler through the packaged Vue compiler.
 * Input JSON contains {handler, vue}; no browser, credentials or network needed.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const scope = vm.createContext({console, setTimeout, clearTimeout});
vm.runInContext(input.vue, scope);

async function exercise({origin='https://ndip-test.ornl.gov', user='owner', privateOK=true, prepared=true}={}) {
  const calls=[];
  const browser={URL, location:{origin:'https://ndip-test.ornl.gov'}, fetch:async (path,options)=>{
    assert.equal(options.credentials,'same-origin');
    assert.equal(options.redirect,'error');
    assert.equal(options.cache,'no-store');
    calls.push(path);
    if(path==='/api/users/current')return {ok:true,json:async()=>({id:user})};
    assert.equal(path,'/history/make_private?history_id=report%20%2F%2B');
    return {ok:privateOK,json:async()=>({message:'private'})};
  }};
  const context={window:browser, URL:undefined, fetch:undefined, trigger:async name=>{
    calls.push(name);
    return name==='radar_contact_prepare' && prepared ? {owner_id:'owner',origin,history_id:'report /+'}:null;
  }};
  const render=scope.Vue.compile('<button @click="'+input.handler+'">Send</button>');
  await render(context,{}).props.onClick();
  return calls;
}

(async()=>{
  assert.deepEqual(await exercise(),['radar_contact_prepare','/api/users/current',
    '/history/make_private?history_id=report%20%2F%2B','radar_contact_send']);
  assert.deepEqual(await exercise({origin:'https://another.example'}),
    ['radar_contact_prepare','radar_contact_privacy_failed']);
  assert.deepEqual(await exercise({user:'another-user'}),
    ['radar_contact_prepare','/api/users/current','radar_contact_privacy_failed']);
  assert.deepEqual(await exercise({privateOK:false}),['radar_contact_prepare','/api/users/current',
    '/history/make_private?history_id=report%20%2F%2B','radar_contact_privacy_failed']);
  assert.deepEqual(await exercise({prepared:false}),['radar_contact_prepare']);
  console.log('5 packaged-Vue feedback privacy-handshake checks passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
