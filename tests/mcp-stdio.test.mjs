import test from "node:test";
import assert from "node:assert/strict";
import {spawn} from "node:child_process";
import {createServer} from "node:http";
import {once} from "node:events";
import {fileURLToPath} from "node:url";
import {resolve,dirname} from "node:path";

const root=resolve(dirname(fileURLToPath(import.meta.url)),"..");
test("local MCP exposes only bounded read-only status and cached traces",async()=>{
  const seen=[];
  const web=createServer((req,res)=>{
    seen.push(req.url);
    res.writeHead(200,{"Content-Type":"application/json"});
    res.end(JSON.stringify(req.url==="/api/status"?{ok:true,paused:false,pop:24}:{ok:true,trace:{input:[0]}}));
  });
  web.listen(0,"127.0.0.1");await once(web,"listening");
  const child=spawn(process.execPath,["src/mcp/stdio.mjs"],{
    cwd:root,env:{...process.env,LAYER1_PORT:String(web.address().port)},stdio:["pipe","pipe","pipe"]});
  let buffer="",waiters=[];
  child.stdout.on("data",chunk=>{
    buffer+=chunk.toString();
    let at;while((at=buffer.indexOf("\n"))>=0){
      const reply=JSON.parse(buffer.slice(0,at));buffer=buffer.slice(at+1);
      const waiter=waiters.shift();if(waiter)waiter(reply);
    }
  });
  let seq=1;
  const rpc=(method,params={})=>new Promise((resolve,reject)=>{
    const id=seq++;
    const t=setTimeout(()=>reject(new Error("MCP timeout")),3000);
    waiters.push(reply=>{clearTimeout(t);assert.equal(reply.id,id);resolve(reply);});
    child.stdin.write(JSON.stringify({jsonrpc:"2.0",id,method,params})+"\n");
  });
  try {
    const init=await rpc("initialize",{protocolVersion:"2025-06-18"});
    assert.equal(init.result.serverInfo.name,"Layer1 Read-only MCP");
    const catalog=await rpc("tools/list");
    assert.deepEqual(catalog.result.tools.map(t=>t.name),["layer1_status_v1","layer1_brain_trace_v1"]);
    const status=await rpc("tools/call",{name:"layer1_status_v1",arguments:{}});
    assert.equal(JSON.parse(status.result.content[0].text).pop,24);
    const trace=await rpc("tools/call",{name:"layer1_brain_trace_v1",arguments:{agent_id:17}});
    assert.equal(trace.result.isError,false);
    const rejected=await rpc("tools/call",{name:"layer1_reset",arguments:{}});
    assert.equal(rejected.error.code,-32602);
    const malicious=await rpc("tools/call",{name:"layer1_brain_trace_v1",arguments:{agent_id:"17/../../api/control"}});
    assert.equal(malicious.error.code,-32602);
    assert.deepEqual(seen,["/api/status","/api/brain/17"]);
  }finally{
    child.kill("SIGTERM");web.close();await once(web,"close");
  }
});
