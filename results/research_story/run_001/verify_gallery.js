// Exercise search, pagination and curated filtering without remote services.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const file=process.argv[2],html=fs.readFileSync(file,'utf8');
const inventory=JSON.parse(html.match(/<script type="application\/json" id="inventory">([\s\S]*?)<\/script>/)[1]);
const script=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0][1];
class Element{
 constructor(){this.value='';this.textContent='';this.children=[];this.events={};this.disabled=false;this.hidden=false;this.dataset={};}
 append(x){this.children.push(x)} replaceChildren(...items){this.children=items}
 addEventListener(type,fn){this.events[type]=fn}
}
const ids={};for(const id of ['inventory','search','experiment','scope','mode','rows','count','prev','next','curatedFilter'])ids[id]=new Element();
ids.inventory.textContent=JSON.stringify(inventory);
const groups=['데이터','Box–Cox','ANN 다중 스텝'].map(group=>{const e=new Element();e.dataset.group=group;return e});
const context={document:{getElementById:id=>ids[id],createElement:()=>new Element(),querySelectorAll:()=>groups}};
vm.runInNewContext(script,context);
assert.equal(ids.rows.children.length,40);assert(ids.count.textContent.includes('1283'));assert(ids.prev.disabled);
ids.next.onclick();assert(ids.count.textContent.includes('41–80'));assert(!ids.prev.disabled);
ids.search.value='IC39';ids.search.events.input();const matching=inventory.filter(r=>r.path.toLowerCase().includes('ic39'));assert.equal(ids.rows.children.length,Math.min(40,matching.length));assert(ids.count.textContent.includes(String(matching.length)));
ids.search.value='';ids.scope.value='main 기준 자료';ids.mode.value='raw';ids.mode.events.input();assert.equal(ids.rows.children.length,Math.min(40,inventory.filter(r=>r.scope===ids.scope.value&&r.mode==='raw').length));
ids.search.value='no-such-file-RESEARCH-000';ids.search.events.input();assert.equal(ids.rows.children.length,0);assert.equal(ids.count.textContent,'검색 결과 없음');
ids.curatedFilter.value='Box–Cox';ids.curatedFilter.onchange();assert(groups[0].hidden);assert(!groups[1].hidden);assert(groups[2].hidden);
process.stdout.write(JSON.stringify({status:'passed',indexed_files:inventory.length,tests:['initial page','next page','path search','scope and mode filters','empty results','curated group filter'],method:'Node VM with a minimal DOM; no browser visual claim'})+'\n');
