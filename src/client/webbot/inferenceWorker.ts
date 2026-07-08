/**
 * Dedicated Web Worker for ONNX inference (see onnxSession.ts).
 *
 * At full map resolution (e.g. the World map's 2000x1000 tile grid) a single
 * AE-encode + policy-forward pass takes multiple seconds in single-threaded
 * WASM. Running that on the main thread would starve Transport's WebSocket
 * ping timer (Transport.ts sends one every 5s) long enough for the server to
 * time the connection out mid-game. Running it here instead keeps the main
 * thread free to keep pinging and processing turns while a decision computes
 * in the background.
 */
import { WebBotModels } from "./onnxSession";
import type { PolicyOutputs } from "./onnxSession";

interface LoadRequest {
  type: "load";
  baseUrl?: string;
}
interface EncodeRequest {
  type: "encode";
  id: number;
  owners: BigInt64Array;
  terrain: Float32Array;
  staticPlanes: Float32Array;
  hr: number;
  wr: number;
  gh: number;
  gw: number;
}
interface PolicyRequest {
  type: "policyForward";
  id: number;
  grid: Float32Array;
  gridValid: Float32Array;
  local: Float32Array;
  players: Float32Array;
  pmask: Float32Array;
  scalars: Float32Array;
  legalActions: Float32Array;
  legalBuild: Float32Array;
  legalNuke: Float32Array;
  legalTile: Float32Array;
  gh: number;
  gw: number;
}
type InRequest = LoadRequest | EncodeRequest | PolicyRequest;

const models = new WebBotModels();

function post(msg: unknown, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(msg, transfer);
}

self.onmessage = async (ev: MessageEvent<InRequest>) => {
  const msg = ev.data;
  const id = "id" in msg ? msg.id : undefined;
  try {
    if (msg.type === "load") {
      await models.load(msg.baseUrl);
      post({ type: "loaded" });
      return;
    }
    if (msg.type === "encode") {
      const z = await models.encode(
        msg.owners,
        msg.terrain,
        msg.staticPlanes,
        msg.hr,
        msg.wr,
        msg.gh,
        msg.gw,
      );
      post({ type: "encode-result", id, z }, [z.buffer]);
      return;
    }
    if (msg.type === "policyForward") {
      const out: PolicyOutputs = await models.policyForward(msg);
      post(
        { type: "policyForward-result", id, ...out },
        [
          out.action.buffer,
          out.player.buffer,
          out.tile.buffer,
          out.build.buffer,
          out.nuke.buffer,
          out.quantity.buffer,
        ],
      );
      return;
    }
  } catch (err) {
    post({ type: "error", id, message: err instanceof Error ? err.message : String(err) });
  }
};
