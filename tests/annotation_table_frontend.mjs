// Run with Node.js: node tests/annotation_table_frontend.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../scMM/components/annotation_table/index.html', import.meta.url), 'utf8');
class Element {
  constructor() { this.style={};this.children=[];this.scrollTop=0;this.scrollLeft=0;this.clientHeight=520;this.listeners={}; }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children=[]; }
  addEventListener(name, callback) { this.listeners[name]=callback; }
  setAttribute(name, value) { this[name]=value; }
}
const elements=Object.fromEntries(['viewport','space','cells','status'].map(name=>[name,new Element()]));
const listeners={}, messages=[];
const context=vm.createContext({
  document:{getElementById:name=>elements[name],createElement:()=>new Element(),querySelector:()=>null},
  window:{addEventListener:(name,callback)=>listeners[name]=callback},
  parent:{postMessage:message=>messages.push(message)},
  setTimeout:fn=>{fn();return 1},clearTimeout:()=>{},
});
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const columns=[{key:'feature_id',label:'ID',width:135},{key:'accepted',label:'Accept',width:85},
 {key:'name',label:'Name',width:230,editable:true},{key:'mirror',label:'Mirror',width:125}];
function render(row) {
 const records=Array.from({length:100},(_,i)=>({annotation_id:'a'+(row+i),feature_id:'f'+(row+i),
  name:'Example',accepted:false,is_isotope:false,has_msms:true,reference_msms_available:true,library_id:'ref',match_level:'MS1'}));
 listeners.message({data:{type:'streamlit:render',args:{dataset:'test',total:100000,row,columns,records}}});
}
render(0);
assert(elements.cells.children.length<100,'Only visible rows should create DOM cells');
elements.cells.children[0].onclick();
assert.equal(messages.at(-1).value.action,'select');
assert.equal(messages.at(-1).value.id,'a0');
const checkbox=elements.cells.children[1].children[0];
checkbox.checked=true;checkbox.onchange();
assert.equal(messages.at(-1).value.id,'a0');
assert.equal(messages.at(-1).value.value,true);
const input=elements.cells.children[2].children[0];
input.value='Edited';input.onchange();
assert.equal(messages.at(-1).value.column,'name');
assert.equal(messages.at(-1).value.value,'Edited');
elements.cells.children[3].children[0].onclick();
assert.equal(messages.at(-1).value.action,'mirror');
elements.viewport.scrollTop=200*34;
elements.viewport.listeners.scroll();
assert.equal(messages.at(-1).value.action,'viewport');
assert.equal(messages.at(-1).value.row,200);
render(200);
assert.equal(elements.cells.children[0].textContent,'f200');
assert(elements.cells.children.length<100);
console.log('Virtual annotation table: bounded DOM, row requests, edits, acceptance and mirror events passed.');
