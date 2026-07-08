/**
 * onnxruntime-web wrapper: loads the two exported graphs (see
 * scripts/export_onnx.py) and runs one decision's worth of inference. CPU
 * (wasm) execution provider only - the whole point is this runs without a
 * server GPU; wasm is also the most broadly-compatible backend across
 * browsers (WebGPU is opt-in future work, see webbot/README notes).
 */
// Wasm-only subpath (not the default "onnxruntime-web" bundle): the default
// bundle also carries WebGPU/JSEP glue and pulls in the much larger
// ort-wasm-simd-threaded.jsep.wasm even when only the "wasm" EP is used.
import * as ort from "onnxruntime-web/wasm";
// Explicit Vite asset-URL imports rather than env.wasm.wasmPaths string
// prefix: onnxruntime-web's wasmPaths-prefix path loads the emscripten glue
// via a plain runtime `import(prefix + filename)`, which Vite dev's import
// analysis mishandles (breaks with "Failed to fetch dynamically imported
// module"). Resolving to a hashed/query URL through the bundler instead
// sidesteps that entirely - see microsoft/onnxruntime PR #26394.
import mjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import { assetUrl } from "../../core/AssetUrls";
import { N_ACTIONS, NUM_STATIC } from "./constants";

ort.env.wasm.wasmPaths = { mjs: mjsUrl, wasm: wasmUrl };
// Threaded wasm needs cross-origin isolation (SharedArrayBuffer) that dev/
// prod hosting doesn't guarantee; single-threaded mode uses the same binary
// without requiring COOP/COEP headers, at the cost of some speed we don't
// need at one decision/second.
ort.env.wasm.numThreads = 1;

export interface PolicyOutputs {
  action: Float32Array; // (N_ACTIONS,) logits
  player: Float32Array; // (MAX_SLOTS,) logits, unmasked by ptarget
  tile: Float32Array; // (gh*gw,) logits, native resolution
  build: Float32Array;
  nuke: Float32Array;
  quantity: Float32Array; // (2,) raw alpha/beta params
  value: number;
}

export class WebBotModels {
  private aeSession: ort.InferenceSession | null = null;
  private policySession: ort.InferenceSession | null = null;
  loaded = false;

  async load(baseUrl: string = assetUrl("webbot/models")): Promise<void> {
    const opts: ort.InferenceSession.SessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    };
    [this.aeSession, this.policySession] = await Promise.all([
      ort.InferenceSession.create(`${baseUrl}/ae_encoder.onnx`, opts),
      ort.InferenceSession.create(`${baseUrl}/policy.onnx`, opts),
    ]);
    this.loaded = true;
  }

  /** AE encoder: (owners int64, terrain float, static float) -> z latent. */
  async encode(
    owners: BigInt64Array,
    terrain: Float32Array,
    staticPlanes: Float32Array,
    hr: number,
    wr: number,
    gh: number,
    gw: number,
  ): Promise<Float32Array> {
    if (!this.aeSession) throw new Error("WebBotModels.load() not called");
    const feeds = {
      owners: new ort.Tensor("int64", owners, [1, hr, wr]),
      terrain: new ort.Tensor("float32", terrain, [1, 3, hr, wr]),
      static: new ort.Tensor("float32", staticPlanes, [1, NUM_STATIC, gh, gw]),
    };
    const out = await this.aeSession.run(feeds);
    return out.z.data as Float32Array;
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
    if (!this.policySession) throw new Error("WebBotModels.load() not called");
    const { gh, gw } = inputs;
    const feeds = {
      grid: new ort.Tensor("float32", inputs.grid, [1, inputs.grid.length / (gh * gw), gh, gw]),
      grid_valid: new ort.Tensor("float32", inputs.gridValid, [1, gh, gw]),
      local: new ort.Tensor(
        "float32",
        inputs.local,
        [1, inputs.local.length / (64 * 64), 64, 64],
      ),
      players: new ort.Tensor("float32", inputs.players, [1, 128, 12]),
      pmask: new ort.Tensor("float32", inputs.pmask, [1, 128]),
      scalars: new ort.Tensor("float32", inputs.scalars, [1, inputs.scalars.length]),
      legal_actions: new ort.Tensor("float32", inputs.legalActions, [1, N_ACTIONS]),
      legal_build: new ort.Tensor("float32", inputs.legalBuild, [1, inputs.legalBuild.length]),
      legal_nuke: new ort.Tensor("float32", inputs.legalNuke, [1, inputs.legalNuke.length]),
      legal_tile: new ort.Tensor("float32", inputs.legalTile, [1, gh, gw]),
    };
    const out = await this.policySession.run(feeds);
    return {
      action: out.action_logits.data as Float32Array,
      player: out.player_logits.data as Float32Array,
      tile: out.tile_logits.data as Float32Array,
      build: out.build_logits.data as Float32Array,
      nuke: out.nuke_logits.data as Float32Array,
      quantity: out.quantity_params.data as Float32Array,
      value: (out.value.data as Float32Array)[0],
    };
  }
}
