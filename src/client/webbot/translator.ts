/**
 * Port of rl/ppo_translate.py's IntentTranslator: choice dict -> engine
 * intent JSON. Region pointers snap to a tile inside the REGION x REGION
 * block that passes the same cheap validity checks the engine runs at
 * execution time (ownership, passability, shore for ports), so region
 * picks rarely become silently-discarded intents.
 */
import { Game } from "../../core/game/Game";
import { getSpawnTiles } from "../../core/execution/Util";
import {
  ACTIONS,
  BUILD_TYPES,
  GH_MAX,
  GW_MAX,
  IMPASSABLE_MAGNITUDE,
  NUKE_TYPES,
  REGION,
} from "./constants";
import { WebBotFeaturizer } from "./features";
import { Choice, Entities, Legal } from "./types";

function randInt(n: number): number {
  return Math.floor(Math.random() * n);
}

export class IntentTranslator {
  private passable: Uint8Array;
  private gh: number;
  private gw: number;
  private width: number;

  constructor(
    private game: Game,
    private builder: WebBotFeaturizer,
  ) {
    const { hr, wr } = builder;
    this.width = game.width();
    const n = hr * wr;
    const passable = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      passable[i] = builder.land[i] === 1 && builder.mag[i] < IMPASSABLE_MAGNITUDE ? 1 : 0;
    }
    this.passable = passable;
    this.gh = hr / REGION;
    this.gw = wr / REGION;
  }

  /** Pick a random valid tile inside the region (engine tile index
   * y*width+x); null if the region has no valid tile at all. span=2 widens
   * the search to a 2x2 region block: coarse-head picks (v7 actions outside
   * REFINE_TILE - boat/nuke/warship) index the top-left /8 region of a /16
   * coarse cell, and the intended target is the whole cell (mirrors
   * IntentTranslator.region_tile in rl/ppo_translate.py). */
  private regionTile(region: number, valid: Uint8Array, span = 1): number | null {
    const stride = Math.max(GW_MAX, this.gw);
    const gy = Math.floor(region / stride);
    const gx = region % stride;
    if (gy >= this.gh || gx >= this.gw) return null; // padded region
    const wr = this.builder.wr;
    const hr = this.builder.hr;
    const candidates: number[] = [];
    for (let dy = 0; dy < REGION * span; dy++) {
      const y = gy * REGION + dy;
      if (y >= hr) break;
      for (let dx = 0; dx < REGION * span; dx++) {
        const x = gx * REGION + dx;
        if (x >= wr) break;
        if (valid[y * wr + x]) candidates.push(y * wr + x);
      }
    }
    if (candidates.length === 0) return null;
    const idx = candidates[randInt(candidates.length)];
    const y = Math.floor(idx / wr);
    const x = idx % wr;
    return y * this.width + x; // engine tile ref uses the full map width
  }

  // classmap: 0 neutral/unowned, 1 own, 2 ally, 3 enemy (see features.ts) -
  // equivalent to Python's slot-based owners==me/ally checks since clut is
  // exactly this same own/ally/enemy/neutral partition.
  private spawnTile(_region: number, _classmap: Uint8Array): number | null {
    // TEMP: BC spawn head is untrained and Asia+400 bots fills land before
    // a policy region pick can land. Mirror SpawnExecution's random path.
    return this.randomEngineSpawnTile();
  }

  /** Same criteria as SpawnExecution.getSpawn(undefined): land, unowned,
   * non-border, and a contiguous spawn blob (requireAllValid). Public so
   * PlaySession can send a spawn intent instantly, without waiting on the
   * (slow, async) ONNX policy pass. */
  randomEngineSpawnTile(): number | null {
    const mg = this.game;
    const w = mg.width();
    const h = mg.height();
    for (let tries = 0; tries < 1000; tries++) {
      const tile = mg.ref(randInt(w), randInt(h));
      if (
        !mg.isLand(tile) ||
        mg.hasOwner(tile) ||
        mg.isBorder(tile) ||
        mg.isImpassable(tile)
      ) {
        continue;
      }
      if (getSpawnTiles(mg.map(), tile, true) !== null) return tile;
    }
    // Looser fallback if the map is nearly full.
    for (let tries = 0; tries < 1000; tries++) {
      const tile = mg.ref(randInt(w), randInt(h));
      if (!mg.isLand(tile) || mg.hasOwner(tile) || mg.isImpassable(tile)) continue;
      if (getSpawnTiles(mg.map(), tile, false).length > 0) return tile;
    }
    return null;
  }

  private boatTile(region: number, classmap: Uint8Array): number | null {
    const valid = new Uint8Array(this.passable.length);
    for (let i = 0; i < valid.length; i++) {
      const c = classmap[i];
      valid[i] = this.passable[i] && c !== 1 && c !== 2 ? 1 : 0;
    }
    // The engine resolves boat destinations via targetTransportTile ->
    // closestShore(owner, dst, 50): a shoreline candidate always resolves,
    // an inland one only if its owner has shore within 50 tiles by land.
    // Prefer shore, fall back to any valid tile (mirrors ppo_translate.py).
    const shoreValid = new Uint8Array(valid.length);
    for (let i = 0; i < valid.length; i++) {
      shoreValid[i] = valid[i] && this.builder.shore[i] === 1 ? 1 : 0;
    }
    const tile = this.regionTile(region, shoreValid, 2);
    return tile !== null ? tile : this.regionTile(region, valid, 2);
  }

  private buildTile(region: number, classmap: Uint8Array, unit: string): number | null {
    if (unit === "Warship") {
      const water = new Uint8Array(this.builder.land.length);
      for (let i = 0; i < water.length; i++) water[i] = this.builder.land[i] === 0 ? 1 : 0;
      return this.regionTile(region, water);
    }
    const valid = new Uint8Array(this.passable.length);
    for (let i = 0; i < valid.length; i++) {
      valid[i] = this.passable[i] && classmap[i] === 1 ? 1 : 0;
    }
    if (unit === "Port") {
      for (let i = 0; i < valid.length; i++) {
        if (valid[i] && this.builder.shore[i] !== 1) valid[i] = 0;
      }
    }
    return this.regionTile(region, valid);
  }

  private slotToPid(slot: number, ents: Entities, lut: Uint8Array): string | null {
    for (const p of ents.players) {
      if (lut[p.id] === slot) return p.pid;
    }
    return null;
  }

  private regionCenter(region: number): [number, number] {
    const stride = Math.max(GW_MAX, this.gw);
    const gy = Math.floor(region / stride);
    const gx = region % stride;
    return [gy * REGION + REGION / 2, gx * REGION + REGION / 2];
  }

  private nearestOwnUnit(
    region: number,
    ents: Entities,
    me: number,
    ids: number[],
    types?: Set<string>,
  ): Entities["units"][number] | null {
    const idset = new Set(ids);
    const cands = ents.units.filter(
      (u) => u.owner === me && idset.has(u.uid) && (!types || types.has(u.type)),
    );
    if (cands.length === 0) return null;
    const [cy, cx] = this.regionCenter(region);
    let best = cands[0];
    let bestD = (best.y - cy) ** 2 + (best.x - cx) ** 2;
    for (const u of cands.slice(1)) {
      const d = (u.y - cy) ** 2 + (u.x - cx) ** 2;
      if (d < bestD) {
        best = u;
        bestD = d;
      }
    }
    return best;
  }

  translate(
    choice: Choice,
    ents: Entities,
    legal: Legal,
    lut: Uint8Array,
    classmap: Uint8Array,
    me: number,
  ): Record<string, unknown>[] {
    const name = ACTIONS[choice.action];
    const a = legal.actions;
    const troops = a.troops ?? 0;
    const frac = Math.min(1.0, Math.max(0.01, choice.quantityFrac ?? 0.25));

    if (name === "noop") return [];
    if (name === "spawn") {
      // Belt-and-suspenders: SpawnExecution re-rolls if we already placed.
      const meP = ents.players.find((p) => p.id === me);
      if (meP && meP.tiles > 0) return [];
      const tile = this.spawnTile(choice.tileRegion ?? -1, classmap);
      if (tile === null) {
        console.warn("[webbot] spawn: no valid tile (map too full?)");
        return [];
      }
      console.log(`[webbot] spawn tile=${tile}`);
      return [{ type: "spawn", tile }];
    }
    if (name === "expand") {
      return [{ type: "attack", targetID: null, troops: Math.floor(troops * frac) }];
    }
    if (name === "attack") {
      const pid = this.slotToPid(choice.playerSlot ?? -1, ents, lut);
      if (pid === null) return [];
      return [{ type: "attack", targetID: pid, troops: Math.floor(troops * frac) }];
    }
    if (name === "boat") {
      const tile = this.boatTile(choice.tileRegion ?? -1, classmap);
      if (tile === null) return [];
      return [{ type: "boat", dst: tile, troops: Math.floor(troops * frac) }];
    }
    if (name === "build") {
      const unit = BUILD_TYPES[choice.buildType ?? 0];
      const tile = this.buildTile(choice.tileRegion ?? -1, classmap, unit);
      if (tile === null) return [];
      return [{ type: "build_unit", unit, tile }];
    }
    if (name === "launch_nuke") {
      const tile = this.regionTile(choice.tileRegion ?? -1, this.passable, 2);
      if (tile === null) return [];
      const [unit, up] = NUKE_TYPES[choice.nukeType ?? 0];
      const intent: Record<string, unknown> = { type: "build_unit", unit, tile };
      if (up !== null) intent.rocketDirectionUp = up;
      return [intent];
    }
    if (name === "retreat") {
      const attacks = a.attacks ?? [];
      if (attacks.length === 0) return [];
      const slot = choice.playerSlot ?? -1;
      let aid = attacks[attacks.length - 1];
      const matches = ents.attacks.filter(
        (atk) =>
          atk.from === me &&
          !atk.retreating &&
          (atk.to ? lut[atk.to] : 0) === slot,
      );
      if (matches.length > 0) aid = matches[matches.length - 1].aid;
      return [{ type: "cancel_attack", attackID: aid }];
    }
    if (name === "upgrade_structure") {
      const u = this.nearestOwnUnit(choice.tileRegion ?? -1, ents, me, a.upgradable ?? []);
      if (u === null) return [];
      return [{ type: "upgrade_structure", unit: u.type, unitId: u.uid }];
    }
    if (name === "move_warship") {
      const ids = a.warships ?? [];
      const water = new Uint8Array(this.builder.land.length);
      for (let i = 0; i < water.length; i++) water[i] = this.builder.land[i] === 0 ? 1 : 0;
      const tile = this.regionTile(choice.tileRegion ?? -1, water, 2);
      if (ids.length === 0 || tile === null) return [];
      return [{ type: "move_warship", unitIds: ids, tile }];
    }
    if (name === "cancel_boat") {
      const u = this.nearestOwnUnit(choice.tileRegion ?? -1, ents, me, a.boats ?? []);
      if (u === null) return [];
      return [{ type: "cancel_boat", unitID: u.uid }];
    }
    if (name === "delete_unit") {
      const u = this.nearestOwnUnit(choice.tileRegion ?? -1, ents, me, a.deletable ?? []);
      if (u === null) return [];
      return [{ type: "delete_unit", unitId: u.uid }];
    }

    const pid = this.slotToPid(choice.playerSlot ?? -1, ents, lut);
    if (pid === null) return [];
    if (name === "alliance_request") return [{ type: "allianceRequest", recipient: pid }];
    if (name === "alliance_reject") return [{ type: "allianceReject", requestor: pid }];
    if (name === "break_alliance") return [{ type: "breakAlliance", recipient: pid }];
    if (name === "donate_gold") {
      const gold = Math.floor(Number(a.gold ?? 0) * frac);
      return [{ type: "donate_gold", recipient: pid, gold }];
    }
    if (name === "donate_troops") {
      return [{ type: "donate_troops", recipient: pid, troops: Math.floor(troops * frac) }];
    }
    if (name === "embargo") return [{ type: "embargo", targetID: pid, action: "start" }];
    if (name === "embargo_stop") return [{ type: "embargo", targetID: pid, action: "stop" }];
    if (name === "target_player") return [{ type: "targetPlayer", target: pid }];
    if (name === "alliance_extension") return [{ type: "allianceExtension", recipient: pid }];
    return [];
  }
}
