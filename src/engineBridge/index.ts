/** Standalone neural action service: starts NO artificial-life world simulation. */
import { startEngineBridge } from "./server.js";

const raw = process.env.ENGINE_BRIDGE_PORT ?? "9090";
const port = Number(raw);
const bridge = startEngineBridge({ port });
bridge.on("listening", () => console.log(`Layer 1+ Engine Bridge available at ws://127.0.0.1:${port}/engine/ws`));
bridge.on("error", (err) => {
  console.error("Cannot start Engine Bridge:", err);
  process.exitCode = 1;
});
