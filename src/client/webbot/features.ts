/**
 * Browser-side port of `rust/ofcore/src/feat.rs::featurize()` (v7 schema,
 * commit e5ee50b - the last commit before the V11 recurrent/entity-obs
 * migration) plus the non-AE parts of the same generation's `encode_grids`
 * (ego ownership pooling, defense-bonus pooling, the local owner-map crop).
 * The AE encode itself (owner embedding + conv stem) runs through the
 * exported ONNX graph - see onnxSession.ts; everything here is plain array
 * math so it can run every decision tick without a ML runtime.
 *
 * Keep this numerically identical to the Rust featurizer: the frozen AE and
 * policy (ppo_v81) were trained on exactly this tensor layout.
 */
import { Game } from "../../core/game/Game";
import {
  ACTIONS,
  BUILD_TYPES,
  DEFENSE_BONUS_BIT,
  FALLOUT_BIT,
  IMPASSABLE_MAGNITUDE,
  IS_LAND_BIT,
  LOCAL,
  MAGNITUDE_MASK,
  MAGNITUDE_NORM,
  MAX_SLOTS,
  N_ACTIONS,
  N_LOCAL,
  N_SCALARS,
  N_TRANSIENT,
  NUKE_TYPES,
  NUKE_UNITS,
  NUM_STATIC,
  OWNER_MASK,
  P_FEAT,
  REGION,
  SHORELINE_BIT,
  STATIC_INDICES,
  TR_ATTACK_RETREAT,
  TR_ATTACK_SRC,
  TR_CONSTRUCTION,
  TR_MIRV_WARHEAD,
  TR_MIRV_WARHEAD_IMPACT,
  TR_NUKE,
  TR_NUKE_IMPACT,
  TR_NUKE_SAMLOCK,
  TR_SAM_COOLDOWN,
  TR_SAM_MISSILE,
  TR_SAM_MISSILE_IMPACT,
  TR_SILO_COOLDOWN,
  TR_STATION,
  TR_TRADE,
  TR_TRADE_DEST,
  TR_TRAIN,
  TR_TRANSPORT,
  TR_TRANSPORT_DEST,
  TR_WARSHIP,
  UNIT_CLASS_INDEX,
  logNorm,
} from "./constants";
import { entities as buildEntities, legality as buildLegality } from "./obsCore";
import { Entities, Legal } from "./types";

const actionIndex: Record<string, number> = Object.fromEntries(
  ACTIONS.map((a, i) => [a, i]),
);

// Static-structure class index -> Missile Silo / SAM Launcher cooldown TR_* base.
const COOLDOWN_TR: Record<number, number> = {
  [UNIT_CLASS_INDEX["Missile Silo"]]: TR_SILO_COOLDOWN,
  [UNIT_CLASS_INDEX["SAM Launcher"]]: TR_SAM_COOLDOWN,
};

export interface PreparedFrame {
  // AE encoder inputs (full resolution)
  ownersFull: BigInt64Array; // (hr*wr) slot per tile, int64 (ONNX owner_emb index)
  terrainInput: Float32Array; // (3, hr, wr): land, magNorm, fallout
  staticPlanes: Float32Array; // (NUM_STATIC, gh, gw)

  // JS-computed grid components (concatenated with the AE latent by the caller)
  // grid = [AE latent (32), ego (3), defense_bonus (1), transient (53)]
  classmap: Uint8Array; // (hr, wr) ego class: 0 neutral/unowned, 1 own, 2 ally, 3 enemy
  ego: Float32Array; // (3, gh, gw): own/ally/enemy fractions
  defenseBonusPooled: Float32Array; // (1, gh, gw): region-average defense-bonus fraction
  transient: Float32Array; // (N_TRANSIENT, gh, gw)
  local: Float32Array; // (N_LOCAL, LOCAL, LOCAL): own/ally/enemy/land/defense_bonus

  // Policy bypass inputs
  players: Float32Array; // (MAX_SLOTS, P_FEAT)
  pmask: Float32Array; // (MAX_SLOTS)
  scalars: Float32Array; // (N_SCALARS)
  legalActions: Float32Array; // (N_ACTIONS)
  legalPtarget: Float32Array; // (N_ACTIONS, MAX_SLOTS)
  legalBuild: Float32Array; // (BUILD_TYPES.length)
  legalNuke: Float32Array; // (NUKE_TYPES.length)
  legalTile: Float32Array; // (gh, gw) - not consumed by policy.onnx (pruned), kept for translator use

  gh: number;
  gw: number;
  hr: number;
  wr: number;
  meSlot: number;
  spawnPhase: boolean;
  alive: boolean;
  entities: Entities;
  legal: Legal;
  me: number; // smallID, -1 if not present
}

export class WebBotFeaturizer {
  lut: Uint8Array | null = null;
  hr = 0;
  wr = 0;
  width = 0;
  height = 0;
  land!: Uint8Array; // (hr*wr)
  mag!: Uint8Array; // (hr*wr)
  shore!: Uint8Array; // (hr*wr)
  defenseBonus!: Uint8Array; // (hr*wr)
  private terrainStatic!: Float32Array; // (2, hr, wr)

  /** Call once at game start (after the map is known). */
  startGame(game: Game): void {
    this.lut = null;
    this.width = game.width();
    this.height = game.height();
    const hr = this.height - (this.height % REGION);
    const wr = this.width - (this.width % REGION);
    this.hr = hr;
    this.wr = wr;
    this.land = new Uint8Array(hr * wr);
    this.mag = new Uint8Array(hr * wr);
    this.shore = new Uint8Array(hr * wr);
    this.defenseBonus = new Uint8Array(hr * wr);
    this.terrainStatic = new Float32Array(2 * hr * wr);
    for (let y = 0; y < hr; y++) {
      for (let x = 0; x < wr; x++) {
        const ref = y * this.width + x;
        const t = game.terrainByte(ref);
        const isLand = (t >> IS_LAND_BIT) & 1;
        const m = t & MAGNITUDE_MASK;
        const isShore = (t >> SHORELINE_BIT) & 1;
        const idx = y * wr + x;
        this.land[idx] = isLand;
        this.mag[idx] = m;
        this.shore[idx] = isShore;
        this.terrainStatic[idx] = isLand;
        this.terrainStatic[hr * wr + idx] = m / MAGNITUDE_NORM;
      }
    }
  }

  private makeLut(players: { id: number }[]): Uint8Array {
    const ids = players.map((p) => p.id).sort((a, b) => a - b);
    const lut = new Uint8Array(4096);
    let slot = 1;
    for (const id of ids) {
      lut[id] = Math.min(slot, MAX_SLOTS - 1);
      slot++;
    }
    return lut;
  }

  private slotLut(players: { id: number }[], spawnPhase: boolean): Uint8Array {
    // During the spawn phase the roster is still filling in (tribe bots
    // spawn over several ticks): rebuild fresh each step, then freeze on
    // the first post-spawn observation for the rest of the episode.
    if (spawnPhase) {
      return this.lut ?? this.makeLut(players);
    }
    if (this.lut === null) this.lut = this.makeLut(players);
    return this.lut;
  }

  /** Ally slots + expiry tick (mirrors feat.rs's `allies`/`ally_expiry`). */
  private allySlots(
    ents: Entities,
    meSlot: number,
    lut: Uint8Array,
  ): { allies: Set<number>; allyExpiry: Map<number, number> } {
    const allies = new Set<number>();
    const allyExpiry = new Map<number, number>();
    for (const [a, b, expiresAt] of ents.alliances) {
      const sa = lut[a];
      const sb = lut[b];
      if (sa === meSlot) {
        allies.add(sb);
        allyExpiry.set(sb, expiresAt);
      } else if (sb === meSlot) {
        allies.add(sa);
        allyExpiry.set(sa, expiresAt);
      }
    }
    return { allies, allyExpiry };
  }

  private legalTile(
    ownersSlot: Uint8Array,
    needsSpawn: boolean,
    gh: number,
    gw: number,
  ): Float32Array {
    const out = new Float32Array(gh * gw);
    if (!needsSpawn) {
      out.fill(1);
      return out;
    }
    const wr = this.wr;
    for (let gy = 0; gy < gh; gy++) {
      for (let gx = 0; gx < gw; gx++) {
        let any = false;
        for (let dy = 0; dy < REGION && !any; dy++) {
          const y = gy * REGION + dy;
          for (let dx = 0; dx < REGION; dx++) {
            const x = gx * REGION + dx;
            const idx = y * wr + x;
            const valid =
              this.land[idx] === 1 &&
              this.mag[idx] < IMPASSABLE_MAGNITUDE &&
              ownersSlot[idx] === 0;
            if (valid) {
              any = true;
              break;
            }
          }
        }
        out[gy * gw + gx] = any ? 1 : 0;
      }
    }
    return out;
  }

  /** (MAX_SLOTS, P_FEAT) player feature block, mirrors feat.rs's player loop. */
  private playerFeats(
    ents: Entities,
    lut: Uint8Array,
    meSlot: number,
    allies: Set<number>,
    allyExpiry: Map<number, number>,
    tick: number,
  ): { players: Float32Array; pmask: Float32Array } {
    const players = new Float32Array(MAX_SLOTS * P_FEAT);
    const pmask = new Float32Array(MAX_SLOTS);

    const atkBetween = new Map<number, number>();
    const outTroops = new Map<number, number>();
    const inTroops = new Map<number, number>();
    const atkTotal = new Map<number, number>();
    const atkRetreating = new Map<number, number>();
    for (const a of ents.attacks) {
      const sa = lut[a.from];
      const sb = a.to ? lut[a.to] : 0;
      if (sa === meSlot) {
        atkBetween.set(sb, (atkBetween.get(sb) ?? 0) + a.troops);
      } else if (sb === meSlot) {
        atkBetween.set(sa, (atkBetween.get(sa) ?? 0) - a.troops);
      }
      outTroops.set(sa, (outTroops.get(sa) ?? 0) + a.troops);
      if (sb !== 0) inTroops.set(sb, (inTroops.get(sb) ?? 0) + a.troops);
      atkTotal.set(sa, (atkTotal.get(sa) ?? 0) + 1);
      if (a.retreating) atkRetreating.set(sa, (atkRetreating.get(sa) ?? 0) + 1);
    }

    // Target marks: who ego or an ally has painted as a priority target.
    const marked = new Set<number>();
    for (const p of ents.players) {
      const slot = lut[p.id];
      if (slot === meSlot || allies.has(slot)) {
        for (const t of p.targets) marked.add(lut[t]);
      }
    }

    for (const p of ents.players) {
      const slot = lut[p.id];
      if (slot <= 0 || slot >= MAX_SLOTS) continue;
      pmask[slot] = 1;
      const meEmbargoed = p.embargoes.map((e) => lut[e]).includes(meSlot);
      const atk = atkBetween.get(slot) ?? 0;
      const nAtk = atkTotal.get(slot) ?? 0;
      const allyExpiryFrac = allyExpiry.has(slot)
        ? Math.min(Math.max((allyExpiry.get(slot)! - tick) / 3000.0, 0), 1)
        : 0;
      const base = slot * P_FEAT;
      players[base + 0] = p.alive ? 1 : 0;
      players[base + 1] = logNorm(p.troops);
      players[base + 2] = logNorm(Number(p.gold));
      players[base + 3] = logNorm(p.tiles);
      players[base + 4] = p.traitor ? 1 : 0;
      players[base + 5] = allies.has(slot) ? 1 : 0;
      players[base + 6] = meEmbargoed ? 1 : 0;
      players[base + 7] = slot === meSlot ? 1 : 0;
      players[base + 8] = logNorm(Math.abs(atk));
      players[base + 9] = atk > 0 ? 1 : 0;
      players[base + 10] = p.reqsIn.length / 4.0;
      players[base + 11] = p.reqsOut.length / 4.0;
      players[base + 12] = logNorm(outTroops.get(slot) ?? 0);
      players[base + 13] = logNorm(inTroops.get(slot) ?? 0);
      players[base + 14] = nAtk > 0 ? (atkRetreating.get(slot) ?? 0) / nAtk : 0;
      players[base + 15] = allyExpiryFrac;
      players[base + 16] = marked.has(slot) ? 1 : 0;
      players[base + 17] = logNorm(p.troopIncome);
      players[base + 18] = logNorm(Number(p.goldIncome));
      players[base + 19] = p.doomsday ? 1 : 0;
      players[base + 20] = Math.min(Math.max(p.doomsdayTicks / 3000.0, 0), 1);
    }
    return { players, pmask };
  }

  private scalars(
    tick: number,
    spawnPhase: boolean,
    alive: boolean,
    legal: Legal,
    ents: Entities,
    meSlot: number,
    me: number,
  ): Float32Array {
    const a = legal.actions;
    const nAlive = ents.players.filter((p) => p.alive).length;
    const mePlayer = ents.players.find((p) => p.id === me);
    return Float32Array.from([
      tick / 15000.0,
      spawnPhase ? 1 : 0,
      alive ? 1 : 0,
      logNorm(a.troops ?? 0),
      logNorm(Number(a.gold ?? 0)),
      nAlive / 128.0,
      (a.attacks?.length ?? 0) / 8.0,
      meSlot / MAX_SLOTS,
      logNorm(mePlayer?.troopIncome ?? 0),
      logNorm(Number(mePlayer?.goldIncome ?? 0)),
      ents.doomsdayEnabled ? 1 : 0,
    ]);
  }

  private masks(
    ents: Entities,
    legal: Legal,
    lut: Uint8Array,
    spawnPhase: boolean,
    alive: boolean,
    me: number,
    /** Engine SpawnExecution re-rolls on every spawn intent (relinquish +
     * re-place). After the first successful pick we must stop sending spawn
     * or the bot teleports every decision tick and can eventually land on a
     * tile getSpawnTiles rejects, leaving it with 0 tiles. */
    needsSpawn: boolean,
  ): {
    legalActions: Float32Array;
    legalPtarget: Float32Array;
    legalBuild: Float32Array;
    legalNuke: Float32Array;
  } {
    const a = legal.actions;
    const act = new Float32Array(N_ACTIONS);
    const ptarget = new Float32Array(N_ACTIONS * MAX_SLOTS);
    const fill = (name: string, ids: number[] | undefined) => {
      if (ids && ids.length > 0) {
        const ai = actionIndex[name];
        act[ai] = 1;
        for (const id of ids) ptarget[ai * MAX_SLOTS + lut[id]] = 1;
      }
    };
    if (spawnPhase && needsSpawn) {
      act[actionIndex["spawn"]] = 1;
    } else {
      // Already placed (or post-spawn): wait out the spawn timer / play.
      act[actionIndex["noop"]] = 1;
    }
    if (alive && a && !spawnPhase) {
      fill("attack", a.attackable);
      fill("alliance_request", a.allianceRequestable);
      fill("alliance_reject", a.allianceRejectable);
      fill("break_alliance", a.breakable);
      fill("donate_gold", a.donatableGold);
      fill("donate_troops", a.donatableTroops);
      fill("embargo", a.embargoable);
      fill("embargo_stop", a.stopEmbargoable);
      fill("target_player", a.targetable);
      fill("alliance_extension", a.extendable);
      act[actionIndex["expand"]] = a.canExpand ?? true ? 1 : 0;
      act[actionIndex["boat"]] =
        (a.canBoat ?? true) && (a.troops ?? 0) > 100 ? 1 : 0;
      const buildOk = BUILD_TYPES.filter((t) => a.buildableTypes?.includes(t));
      act[actionIndex["build"]] = buildOk.length > 0 ? 1 : 0;
      const nukesOk = NUKE_UNITS.filter((t) => a.buildableTypes?.includes(t));
      act[actionIndex["launch_nuke"]] = nukesOk.length > 0 && a.hasSilo ? 1 : 0;
      if (a.attacks && a.attacks.length > 0) {
        act[actionIndex["retreat"]] = 1;
        const r = actionIndex["retreat"];
        for (const atk of ents.attacks) {
          if (atk.from === me) {
            ptarget[r * MAX_SLOTS + (atk.to ? lut[atk.to] : 0)] = 1;
          }
        }
      }
      act[actionIndex["upgrade_structure"]] =
        a.upgradable && a.upgradable.length > 0 ? 1 : 0;
      act[actionIndex["move_warship"]] =
        a.warships && a.warships.length > 0 ? 1 : 0;
      act[actionIndex["cancel_boat"]] = a.boats && a.boats.length > 0 ? 1 : 0;
      act[actionIndex["delete_unit"]] =
        a.deletable && a.deletable.length > 0 ? 1 : 0;
    }
    const legalBuild = Float32Array.from(
      BUILD_TYPES.map((t) => (alive && a?.buildableTypes?.includes(t) ? 1 : 0)),
    );
    const legalNuke = Float32Array.from(
      NUKE_TYPES.map(([u]) => (alive && a?.buildableTypes?.includes(u) ? 1 : 0)),
    );
    return { legalActions: act, legalPtarget: ptarget, legalBuild, legalNuke };
  }

  /** Ego-class classmap + per-region ownership-fraction pooling (own/ally/
   * enemy) + defense-bonus pooling. Mirrors feat.rs's pool_ego_db. */
  private egoAndClassmap(
    ownersSlot: Uint8Array,
    clut: Uint8Array,
    hr: number,
    wr: number,
    gh: number,
    gw: number,
  ): { classmap: Uint8Array; ego: Float32Array; defenseBonusPooled: Float32Array } {
    const classmap = new Uint8Array(hr * wr);
    for (let i = 0; i < ownersSlot.length; i++) {
      classmap[i] = clut[ownersSlot[i]];
    }
    const ego = new Float32Array(3 * gh * gw);
    const defenseBonusPooled = new Float32Array(gh * gw);
    const cellArea = REGION * REGION;
    for (let gy = 0; gy < gh; gy++) {
      for (let gx = 0; gx < gw; gx++) {
        let own = 0,
          ally = 0,
          enemy = 0,
          db = 0;
        for (let dy = 0; dy < REGION; dy++) {
          const y = gy * REGION + dy;
          const rowBase = y * wr + gx * REGION;
          for (let dx = 0; dx < REGION; dx++) {
            const c = classmap[rowBase + dx];
            if (c === 1) own++;
            else if (c === 2) ally++;
            else if (c === 3) enemy++;
            db += this.defenseBonus[rowBase + dx];
          }
        }
        const gi = gy * gw + gx;
        ego[gi] = own / cellArea;
        ego[gh * gw + gi] = ally / cellArea;
        ego[2 * gh * gw + gi] = enemy / cellArea;
        defenseBonusPooled[gi] = db / cellArea;
      }
    }
    return { classmap, ego, defenseBonusPooled };
  }

  /** LOCAL x LOCAL crop (own/ally/enemy/land/defense_bonus planes) centered
   * on own territory's centroid (map center if the agent owns nothing). */
  private localCrop(classmap: Uint8Array, hr: number, wr: number): Float32Array {
    // Pad up to LOCAL if the map is smaller (mirrors the Rust local_crop's F.pad).
    const H = Math.max(hr, LOCAL);
    const W = Math.max(wr, LOCAL);
    let sumY = 0,
      sumX = 0,
      count = 0;
    for (let y = 0; y < hr; y++) {
      const rowBase = y * wr;
      for (let x = 0; x < wr; x++) {
        if (classmap[rowBase + x] === 1) {
          sumY += y;
          sumX += x;
          count++;
        }
      }
    }
    const cy = count > 0 ? sumY / count : H / 2;
    const cx = count > 0 ? sumX / count : W / 2;
    let y0 = Math.round(cy - LOCAL / 2);
    let x0 = Math.round(cx - LOCAL / 2);
    y0 = Math.min(Math.max(y0, 0), H - LOCAL);
    x0 = Math.min(Math.max(x0, 0), W - LOCAL);

    const out = new Float32Array(N_LOCAL * LOCAL * LOCAL);
    const plane = LOCAL * LOCAL;
    for (let dy = 0; dy < LOCAL; dy++) {
      const y = y0 + dy;
      for (let dx = 0; dx < LOCAL; dx++) {
        const x = x0 + dx;
        let c = 0;
        let land = 0;
        let db = 0;
        if (y < hr && x < wr) {
          const idx = y * wr + x;
          c = classmap[idx];
          land = this.land[idx];
          db = this.defenseBonus[idx];
        }
        const oi = dy * LOCAL + dx;
        out[0 * plane + oi] = c === 1 ? 1 : 0;
        out[1 * plane + oi] = c === 2 ? 1 : 0;
        out[2 * plane + oi] = c === 3 ? 1 : 0;
        out[3 * plane + oi] = land;
        out[4 * plane + oi] = db;
      }
    }
    return out;
  }

  /** Full featurization for one decision step. */
  prepare(game: Game, clientID: string): PreparedFrame {
    const ents = buildEntities(game) as unknown as Entities;
    const legal = buildLegality(game, clientID) as unknown as Legal;
    const spawnPhase = game.inSpawnPhase();
    const lut = this.slotLut(ents.players, spawnPhase);
    const agent = game.playerByClientID(clientID);
    const me = agent ? agent.smallID() : -1;
    const alive = agent ? agent.isAlive() : false;
    // First successful spawn sets hasSpawned(); further spawn intents re-roll.
    const needsSpawn = spawnPhase && !(agent?.hasSpawned() ?? false);
    const meSlot = me >= 0 ? lut[me] : 0;
    const { hr, wr } = this;
    const gh = hr / REGION;
    const gw = wr / REGION;

    const tileState = game.tileStateBuffer();
    const ownersSlot = new Uint8Array(hr * wr); // fast path for JS-side pooling/crops
    const ownersFull = new BigInt64Array(hr * wr); // int64 for the ONNX owner embedding
    const fallout = new Float32Array(hr * wr);
    for (let y = 0; y < hr; y++) {
      const srcBase = y * this.width;
      const dstBase = y * wr;
      for (let x = 0; x < wr; x++) {
        const state = tileState[srcBase + x];
        const smallId = state & OWNER_MASK;
        const slot = lut[smallId];
        ownersSlot[dstBase + x] = slot;
        ownersFull[dstBase + x] = BigInt(slot);
        fallout[dstBase + x] = state & FALLOUT_BIT ? 1 : 0;
        this.defenseBonus[dstBase + x] = state & DEFENSE_BONUS_BIT ? 1 : 0;
      }
    }

    const allies0 = this.allySlots(ents, meSlot, lut);
    const clut = new Uint8Array(MAX_SLOTS).fill(3);
    clut[0] = 0;
    for (const s of allies0.allies) clut[s] = 2;
    if (meSlot > 0) clut[meSlot] = 1;

    const staticPlanes = new Float32Array(NUM_STATIC * gh * gw);
    const transient = new Float32Array(N_TRANSIENT * gh * gw);
    const plane = gh * gw;
    const staticPos = new Map(STATIC_INDICES.map((ci, k) => [ci, k]));
    for (const u of ents.units) {
      const ci = UNIT_CLASS_INDEX[u.type];
      if (ci === undefined) continue;
      const gy = Math.floor(u.y / REGION);
      const gx = Math.floor(u.x / REGION);
      if (!(gy >= 0 && gy < gh && gx >= 0 && gx < gw)) continue;
      const gi = gy * gw + gx;

      const ownerSlot = Math.min(lut[u.owner] ?? 0, MAX_SLOTS - 1);
      const cls = clut[ownerSlot];
      const oc = cls !== 0 ? cls - 1 : 2;

      if (staticPos.has(ci) && !u.constructing) {
        const levelFrac = Math.min(u.level / 10.0, 1.0);
        const idx = staticPos.get(ci)! * plane + gi;
        staticPlanes[idx] = Math.max(staticPlanes[idx], levelFrac);
      }

      let ty = -1,
        tx = -1;
      if (u.tx !== null && u.ty !== null) {
        ty = Math.floor(u.ty / REGION);
        tx = Math.floor(u.tx / REGION);
      }
      const targetOk = ty >= 0 && ty < gh && tx >= 0 && tx < gw;
      const tgi = targetOk ? ty * gw + tx : -1;

      const setMax = (base: number, value: number) => {
        const idx = (base + oc) * plane + gi;
        transient[idx] = Math.max(transient[idx], value);
      };
      const setOne = (base: number, at: number) => {
        transient[(base + oc) * plane + at] = 1;
      };

      switch (u.type) {
        case "Warship": {
          const hf =
            u.health !== null && u.maxHealth !== null && u.maxHealth > 0
              ? u.health / u.maxHealth
              : 1.0;
          setMax(TR_WARSHIP, hf);
          break;
        }
        case "Transport": {
          setMax(TR_TRANSPORT, logNorm(u.troops));
          if (targetOk) setOne(TR_TRANSPORT_DEST, tgi);
          break;
        }
        case "Trade Ship": {
          setOne(TR_TRADE, gi);
          if (targetOk) setOne(TR_TRADE_DEST, tgi);
          break;
        }
        case "Atom Bomb":
        case "Hydrogen Bomb":
        case "MIRV": {
          setOne(TR_NUKE, gi);
          if (targetOk) setOne(TR_NUKE_IMPACT, tgi);
          if (u.samLock) setOne(TR_NUKE_SAMLOCK, gi);
          break;
        }
        case "SAMMissile": {
          setOne(TR_SAM_MISSILE, gi);
          if (targetOk) setOne(TR_SAM_MISSILE_IMPACT, tgi);
          break;
        }
        case "MIRV Warhead": {
          setOne(TR_MIRV_WARHEAD, gi);
          if (targetOk) setOne(TR_MIRV_WARHEAD_IMPACT, tgi);
          break;
        }
        case "Train": {
          setMax(TR_TRAIN, logNorm(u.troops));
          break;
        }
        default:
          break;
      }
      if (u.constructing) setOne(TR_CONSTRUCTION, gi);
      if (u.cooldown) {
        const base = COOLDOWN_TR[ci];
        if (base !== undefined) setOne(base, gi);
      }
      if (u.station) setOne(TR_STATION, gi);
    }

    // Attack fronts: source-tile intensity across ALL active attacks
    // (shared planes, not ego-split, per the v7 schema).
    for (const a of ents.attacks) {
      if (a.srcX === null || a.srcY === null) continue;
      const gy = Math.floor(a.srcY / REGION);
      const gx = Math.floor(a.srcX / REGION);
      if (!(gy >= 0 && gy < gh && gx >= 0 && gx < gw)) continue;
      const gi = gy * gw + gx;
      const val = logNorm(a.troops);
      let idx = TR_ATTACK_SRC * plane + gi;
      transient[idx] = Math.max(transient[idx], val);
      if (a.retreating) {
        idx = TR_ATTACK_RETREAT * plane + gi;
        transient[idx] = Math.max(transient[idx], val);
      }
    }

    const { classmap, ego, defenseBonusPooled } = this.egoAndClassmap(
      ownersSlot,
      clut,
      hr,
      wr,
      gh,
      gw,
    );
    const local = this.localCrop(classmap, hr, wr);
    const { allies, allyExpiry } = allies0;
    const { players, pmask } = this.playerFeats(
      ents,
      lut,
      meSlot,
      allies,
      allyExpiry,
      game.ticks(),
    );
    const scalars = this.scalars(game.ticks(), spawnPhase, alive, legal, ents, meSlot, me);
    const { legalActions, legalPtarget, legalBuild, legalNuke } = this.masks(
      ents,
      legal,
      lut,
      spawnPhase,
      alive,
      me,
      needsSpawn,
    );
    const legalTile = this.legalTile(ownersSlot, needsSpawn, gh, gw);

    const terrainInput = new Float32Array(3 * hr * wr);
    terrainInput.set(this.terrainStatic.subarray(0, hr * wr), 0);
    terrainInput.set(this.terrainStatic.subarray(hr * wr, 2 * hr * wr), hr * wr);
    terrainInput.set(fallout, 2 * hr * wr);

    return {
      ownersFull,
      terrainInput,
      staticPlanes,
      classmap,
      ego,
      defenseBonusPooled,
      transient,
      local,
      players,
      pmask,
      scalars,
      legalActions,
      legalPtarget,
      legalBuild,
      legalNuke,
      legalTile,
      gh,
      gw,
      hr,
      wr,
      meSlot,
      spawnPhase,
      alive,
      entities: ents,
      legal,
      me,
    };
  }
}
