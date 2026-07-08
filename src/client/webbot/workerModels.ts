/**
 * Main-thread stand-in for WebBotModels (onnxSession.ts) that proxies every
 * call into inferenceWorker.ts over postMessage, so the actual (multi-second,
 * at full map resolution) WASM inference never runs on the main thread. See
 * inferenceWorker.ts for why this matters.
 */
import type { PolicyOutputs } from "./onnxSession";

interface PendingEntry {
  resolve: (data: MessageEventData) => void;
  reject: (err: Error) => void;
}

type MessageEventData = Record<string, unknown> & { type: string; id?: number };

// Inlined as a same-origin Blob (Vite's `?worker&inline`), matching
// WorkerClient.ts's game-engine worker: sidesteps the cross-origin
// `new Worker(url)` restriction that would otherwise apply once this bundle
// is served from a CDN.
async function createInferenceWorker(): Promise<Worker> {
  const { default: InferenceWorker } = await import(
    "./inferenceWorker.ts?worker&inline"
  );
  return new InferenceWorker();
}

export class WorkerModels {
  private worker: Worker | null = null;
  private ready: Promise<Worker>;
  private nextId = 0;
  private pending = new Map<number, PendingEntry>();
  private loadPending: PendingEntry | null = null;
  loaded = false;

  constructor() {
    this.ready = createInferenceWorker().then((worker) => {
      this.worker = worker;
      worker.onmessage = (ev: MessageEvent<MessageEventData>) => this.onMessage(ev.data);
      worker.onerror = (ev: ErrorEvent) => {
        const err = new Error(`inference worker error: ${ev.message}`);
        if (this.loadPending) {
          this.loadPending.reject(err);
          this.loadPending = null;
        }
        for (const entry of this.pending.values()) entry.reject(err);
        this.pending.clear();
      };
      return worker;
    });
  }

  private onMessage(data: MessageEventData): void {
    if (data.type === "loaded") {
      this.loaded = true;
      this.loadPending?.resolve(data);
      this.loadPending = null;
      return;
    }
    if (data.type === "error") {
      const err = new Error(String(data.message));
      if (data.id !== undefined && this.pending.has(data.id)) {
        this.pending.get(data.id)!.reject(err);
        this.pending.delete(data.id);
      } else {
        this.loadPending?.reject(err);
        this.loadPending = null;
      }
      return;
    }
    if (data.id === undefined) return;
    const entry = this.pending.get(data.id);
    if (!entry) return;
    this.pending.delete(data.id);
    entry.resolve(data);
  }

  async load(baseUrl?: string): Promise<void> {
    const worker = await this.ready;
    await new Promise<void>((resolve, reject) => {
      this.loadPending = { resolve: () => resolve(), reject };
      worker.postMessage({ type: "load", baseUrl });
    });
  }

  async encode(
    owners: BigInt64Array,
    terrain: Float32Array,
    staticPlanes: Float32Array,
    hr: number,
    wr: number,
    gh: number,
    gw: number,
  ): Promise<Float32Array> {
    const worker = await this.ready;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: (data) => resolve(data.z as Float32Array), reject });
      worker.postMessage(
        { type: "encode", id, owners, terrain, staticPlanes, hr, wr, gh, gw },
        [owners.buffer, terrain.buffer, staticPlanes.buffer],
      );
    });
  }

  async policyForward(inputs: {
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
  }): Promise<PolicyOutputs> {
    const worker = await this.ready;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {
        resolve: (data) =>
          resolve({
            action: data.action as Float32Array,
            player: data.player as Float32Array,
            tile: data.tile as Float32Array,
            build: data.build as Float32Array,
            nuke: data.nuke as Float32Array,
            quantity: data.quantity as Float32Array,
            value: data.value as number,
          }),
        reject,
      });
      worker.postMessage(
        { type: "policyForward", id, ...inputs },
        [
          inputs.grid.buffer,
          inputs.gridValid.buffer,
          inputs.local.buffer,
          inputs.players.buffer,
          inputs.pmask.buffer,
          inputs.scalars.buffer,
          inputs.legalActions.buffer,
          inputs.legalBuild.buffer,
          inputs.legalNuke.buffer,
          inputs.legalTile.buffer,
        ],
      );
    });
  }
}
