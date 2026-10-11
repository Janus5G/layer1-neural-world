import test from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import { WebSocket } from "ws";
import { startEngineBridge } from "../dist/engineBridge/server.js";

async function openHarness(opts = {}) {
  const server = startEngineBridge({ port: 0, ...opts });
  await once(server, "listening");
  const { port, address } = server.address();
  assert.equal(address, "127.0.0.1", "Bridge must be loopback-only");
  const clients = [];
  async function connect() {
    const ws = new WebSocket(`ws://127.0.0.1:${port}/engine/ws`);
    const inbox = [];
    const waiters = [];
    ws.on("message", b => {
      const value = JSON.parse(b.toString());
      const next = waiters.shift();
      if (next) next(value);
      else inbox.push(value);
    });
    await once(ws, "open");
    clients.push(ws);
    return {
      ws,
      async next() {
        if (inbox.length) return inbox.shift();
        return await Promise.race([
          new Promise(resolve => waiters.push(resolve)),
          new Promise((_, reject) => setTimeout(() => reject(new Error("WebSocket response timeout")), 1500)),
        ]);
      },
      async send(payload) { ws.send(JSON.stringify(payload)); return await this.next(); },
    };
  }
  async function close() {
    for (const ws of clients) ws.terminate();
    await new Promise(resolve => server.close(resolve));
  }
  return { connect, close };
}

const create = {type:"create", agentId:"unit-1", inputDim:6, actions:["north","south","east","west","idle"], learn:true};

test("handshake, decisions, feedback training, terminal and remove", async () => {
  const h = await openHarness({ trainingInterval: 3 });
  try {
    const c = await h.connect();
    const hello = await c.next();
    assert.equal(hello.type,"hello"); assert.equal(hello.version,1);
    assert.equal((await c.send(create)).type,"created");
    const observation = [0.1,0.2,-0.2,0.1,0.3,0.8];
    let previous = null;
    for (let i=0;i<5;i++) {
      const res = await c.send({type:"step",agentId:"unit-1",observation,...(i>0?{reward:0.5}:{})});
      assert.equal(res.type,"action");
      assert.ok(create.actions.includes(res.name));
      assert.ok(Number.isFinite(res.value));
      assert.ok(res.index>=0 && res.index<5);
      previous = res;
    }
    assert.ok(previous.updates>=1, "training should have run at least once");
    const ended = await c.send({type:"step",agentId:"unit-1",done:true,reward:0});
    assert.equal(ended.type,"episode_end");
    const restarted = await c.send({type:"step",agentId:"unit-1",observation});
    assert.equal(restarted.type,"action");
    assert.equal((await c.send({type:"remove",agentId:"unit-1"})).type,"removed");
    assert.equal((await c.send({type:"step",agentId:"unit-1",observation})).code,"unknown_agent");
  } finally { await h.close(); }
});

test("reject malformed payloads and bounds without modifying agent state", async () => {
  const h = await openHarness();
  try {
    const c = await h.connect(); await c.next();
    assert.equal((await c.send({type:"create",agentId:"bad",inputDim:9999,actions:["go","stop"]})).code,"invalid_input_dim");
    assert.equal((await c.send({...create, actions:["north","north"]})).code,"invalid_actions");
    assert.equal((await c.send(create)).type,"created");
    assert.equal((await c.send(create)).code,"agent_exists");
    assert.equal((await c.send({type:"step",agentId:"unit-1",observation:[0,0]})).code,"invalid_observation");
    assert.equal((await c.send({type:"step",agentId:"unit-1",observation:[0,0,0,0,0,9]})).code,"invalid_observation");
    assert.equal((await c.send({type:"step",agentId:"unit-1",observation:[0,0,0,0,0,0]})).type,"action");
    assert.equal((await c.send({type:"step",agentId:"unit-1",observation:[0,0,0,0,0,0]})).code,"invalid_reward");
    assert.equal((await c.send({type:"step",agentId:"unit-1",observation:[0,0,0,0,0,0],reward:100})).code,"invalid_reward");
    assert.equal((await c.send({type:"step",agentId:"unit-1",observation:[0,0,0,0,0,0],reward:0.1})).type,"action");
    assert.equal((await c.send({type:"nope"})).code,"unknown_type");
  } finally { await h.close(); }
});

test("each connection owns its own agents and isolated neural state", async () => {
  const h = await openHarness();
  try {
    const a=await h.connect(), b=await h.connect();
    await a.next(); await b.next();
    assert.equal((await a.send(create)).type,"created");
    assert.equal((await b.send({type:"step",agentId:"unit-1",observation:[0,0,0,0,0,0]})).code,"unknown_agent");
    assert.equal((await b.send(create)).type,"created");
  } finally { await h.close(); }
});

test("enforce per-connection agent limit", async () => {
  const h = await openHarness();
  try {
    const c=await h.connect(); await c.next();
    for(let i=0;i<8;i++)assert.equal((await c.send({...create,agentId:`a${i}`})).type,"created");
    assert.equal((await c.send({...create,agentId:"a8"})).code,"agent_limit");
  } finally { await h.close(); }
});

test("cross-site browser Origin headers are rejected", async () => {
  const h = await openHarness();
  try {
    // A rejected upgrade produces an unexpected-response or error, not an open socket.
    const server = await h.connect();
    assert.equal((await server.next()).type, "hello");
    const address = server.ws.url;
    const hostile = new WebSocket(address, { headers: { Origin: "https://evil.example" } });
    const outcome = await Promise.race([
      once(hostile, "unexpected-response").then(() => "rejected"),
      once(hostile, "open").then(() => "opened"),
    ]);
    assert.equal(outcome, "rejected");
    hostile.terminate();
  } finally { await h.close(); }
});
