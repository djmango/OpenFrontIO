/**
 * WebBot: ties together featurization (features.ts), ONNX inference
 * (onnxSession.ts), action sampling (sampling.ts), and intent translation
 * (translator.ts) into the same decide-once-per-N-ticks loop rl/play.py
 * runs, but entirely in the browser - no Python subprocess, no server GPU.
 */
import { Game } from "../../core/game/Game";
import {
  ACTIONS,
  C_GRID,
  GW_MAX,
  LATENT_C,
  MAX_SLOTS,
  NEEDS_PLAYER,
  NEEDS_QUANTITY,
  NEEDS_TILE,
} from "./constants";
import { WebBotFeaturizer } from "./features";
import { betaParams, sampleBeta, sampleCategorical } from "./sampling";
import { IntentTranslator } from "./translator";
import { Choice } from "./types";
import { WorkerModels } from "./workerModels";

export interface WebBotDebugInfo {
  tick: number;
  action: string;
  value: number;
  tiles: number;
  troops: number;
  probs: number[];
}

export interface WebBotOptions {
  greedy?: boolean;
  modelsBaseUrl?: string;
  onDebug?: (info: WebBotDebugInfo) => void;
}

function softmax(logits: Float32Array): number[] {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export class WebBot {
  private featurizer = new WebBotFeaturizer();
  private models = new WorkerModels();
  private translator: IntentTranslator | null = null;
  private greedy: boolean;
  private onDebug: WebBotOptions["onDebug"];

  constructor(private opts: WebBotOptions = {}) {
    this.greedy = opts.greedy ?? false;
    this.onDebug = opts.onDebug;
  }

  async load(): Promise<void> {
    await this.models.load(this.opts.modelsBaseUrl);
  }

  /** Call once when the map/game is known (before the first decide()). */
  startGame(game: Game): void {
    this.featurizer.startGame(game);
    this.translator = new IntentTranslator(game, this.featurizer);
  }

  /** One decision: featurize -> AE encode -> policy forward -> sample ->
   * translate. Returns the engine intents to send this tick. */
  async decide(game: Game, clientID: string): Promise<Record<string, unknown>[]> {
    if (this.translator === null) {
      throw new Error("WebBot.startGame() must run before decide()");
    }
    const frame = this.featurizer.prepare(game, clientID);
    const { gh, gw, hr, wr } = frame;

    const z = await this.models.encode(
      frame.ownersFull,
      frame.terrainInput,
      frame.staticPlanes,
      hr,
      wr,
      gh,
      gw,
    );

    // grid = [AE latent (32ch), ego own/ally/enemy (3ch), transient (8ch)]
    // = C_GRID channels, matching rl/obs.py's encode_grids() concatenation.
    const gridSize = gh * gw;
    const grid = new Float32Array(C_GRID * gridSize);
    grid.set(z, 0);
    grid.set(frame.ego, LATENT_C * gridSize);
    grid.set(frame.transient, (LATENT_C + 3) * gridSize);
    const gridValid = new Float32Array(gridSize).fill(1);

    const out = await this.models.policyForward({
      grid,
      gridValid,
      local: frame.local,
      players: frame.players,
      pmask: frame.pmask,
      scalars: frame.scalars,
      legalActions: frame.legalActions,
      legalBuild: frame.legalBuild,
      legalNuke: frame.legalNuke,
      legalTile: frame.legalTile,
      gh,
      gw,
    });

    const actionIdx = sampleCategorical(out.action, this.greedy);
    const name = ACTIONS[actionIdx];
    const choice: Choice = { action: actionIdx };

    if (NEEDS_PLAYER.has(name)) {
      // Player head mask depends on the sampled action (matches
      // Policy.act(): legal_ptarget.gather(1, sampled_action)).
      const ptarget = frame.legalPtarget.subarray(
        actionIdx * MAX_SLOTS,
        (actionIdx + 1) * MAX_SLOTS,
      );
      const masked = new Float32Array(MAX_SLOTS);
      for (let i = 0; i < MAX_SLOTS; i++) {
        masked[i] = ptarget[i] > 0.5 ? out.player[i] : -1e9;
      }
      choice.playerSlot = sampleCategorical(masked, this.greedy);
    }
    if (NEEDS_TILE.has(name)) {
      const localIdx = sampleCategorical(out.tile, this.greedy);
      // Batch-local flat index -> GW_MAX-stride global region index (see
      // Policy._local_to_global in rl/policy.py).
      const stride = Math.max(GW_MAX, gw);
      choice.tileRegion = Math.floor(localIdx / gw) * stride + (localIdx % gw);
    }
    if (name === "build") {
      choice.buildType = sampleCategorical(out.build, this.greedy);
    }
    if (name === "launch_nuke") {
      choice.nukeType = sampleCategorical(out.nuke, this.greedy);
    }
    if (NEEDS_QUANTITY.has(name)) {
      const [alpha, beta] = betaParams(out.quantity);
      choice.quantityFrac = sampleBeta(alpha, beta, this.greedy);
    }

    if (this.onDebug) {
      const me = game.playerByClientID(clientID);
      this.onDebug({
        tick: game.ticks(),
        action: name,
        value: out.value,
        tiles: me?.numTilesOwned() ?? 0,
        troops: me?.troops() ?? 0,
        probs: softmax(out.action),
      });
    }

    return this.translator.translate(
      choice,
      frame.entities,
      frame.legal,
      this.featurizer.lut!,
      frame.classmap,
      frame.me,
    );
  }
}
