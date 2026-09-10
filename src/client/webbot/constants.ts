/**
 * Constants ported 1:1 from the training pipeline (rust/ofcore/src/feat.rs,
 * rust/oftrain/src/policy.rs @ commit e5ee50b - the last "v7" schema before
 * the V11 recurrent/entity-obs migration). Keep these in sync with the Rust
 * source - they must match exactly for the ONNX-exported AE encoder +
 * policy to see the same tensor layout they were trained on.
 */

export const IS_LAND_BIT = 7;
export const SHORELINE_BIT = 6;
export const MAGNITUDE_MASK = 0x1f;
export const MAGNITUDE_NORM = 31.0;
export const IMPASSABLE_MAGNITUDE = 31;

// Tile-state (GameMap.tileStateBuffer) bit layout, see GameMap.ts.
export const OWNER_MASK = 0xfff;
export const FALLOUT_BIT = 1 << 13;
export const DEFENSE_BONUS_BIT = 1 << 14;

// v7 spatial resolution: one latent cell / tile-pointer region per 8x8 tiles.
export const REGION = 8;
export const LATENT_C = 32;
export const MAX_SLOTS = 128;

// grid = [AE latent (32), ego own/ally/enemy (3), defense_bonus (1),
// transient (53)] = C_GRID 89 (rust/oftrain/src/policy.rs @ e5ee50b).
export const N_TRANSIENT = 53;
export const N_DEFENSE_BONUS = 1;
export const C_GRID = LATENT_C + 3 + N_DEFENSE_BONUS + N_TRANSIENT;

export const LOCAL = 64;
export const N_LOCAL = 5;

export const P_FEAT = 21;
export const N_SCALARS = 11;

// Largest featurized grid across the curriculum (Asia at the 1/8 latent
// resolution) - region indices use this stride regardless of the actual
// map's grid size (see IntentTranslator.region_tile).
export const GH_MAX = 150;
export const GW_MAX = 250;

// Transient plane base offsets (rust/ofcore/src/feat.rs TR_*); each is an
// own/ally/enemy triplet (base + oc, oc in {0,1,2}) except the last two,
// which are single planes shared across owners in the v7 schema.
export const TR_WARSHIP = 0;
export const TR_TRANSPORT = 3;
export const TR_TRANSPORT_DEST = 6;
export const TR_TRADE = 9;
export const TR_TRADE_DEST = 12;
export const TR_NUKE = 15;
export const TR_NUKE_IMPACT = 18;
export const TR_NUKE_SAMLOCK = 21;
export const TR_CONSTRUCTION = 24;
export const TR_SAM_MISSILE = 27;
export const TR_SAM_MISSILE_IMPACT = 30;
export const TR_MIRV_WARHEAD = 33;
export const TR_MIRV_WARHEAD_IMPACT = 36;
export const TR_TRAIN = 39;
export const TR_SILO_COOLDOWN = 42;
export const TR_SAM_COOLDOWN = 45;
export const TR_STATION = 48;
export const TR_ATTACK_SRC = 51; // shared across owners
export const TR_ATTACK_RETREAT = 52; // shared, overlays TR_ATTACK_SRC

export const ACTIONS = [
  "noop",
  "attack",
  "expand",
  "boat",
  "build",
  "launch_nuke",
  "alliance_request",
  "alliance_reject",
  "break_alliance",
  "donate_gold",
  "donate_troops",
  "embargo",
  "retreat",
  "spawn",
  "upgrade_structure",
  "move_warship",
  "cancel_boat",
  "delete_unit",
  "embargo_stop",
  "target_player",
  "alliance_extension",
] as const;
export type ActionName = (typeof ACTIONS)[number];
export const N_ACTIONS = ACTIONS.length;

export const BUILD_TYPES = [
  "City",
  "Port",
  "Defense Post",
  "Missile Silo",
  "SAM Launcher",
  "Factory",
  "Warship",
] as const;

// (engine unit, rocketDirectionUp | null). MIRV ignores the arc flag.
export const NUKE_TYPES: [string, boolean | null][] = [
  ["Atom Bomb", true],
  ["Atom Bomb", false],
  ["Hydrogen Bomb", true],
  ["Hydrogen Bomb", false],
  ["MIRV", null],
];
export const NUKE_UNITS = ["Atom Bomb", "Hydrogen Bomb", "MIRV"];

export const NEEDS_PLAYER = new Set<ActionName>([
  "attack",
  "alliance_request",
  "alliance_reject",
  "break_alliance",
  "donate_gold",
  "donate_troops",
  "embargo",
  "retreat",
  "embargo_stop",
  "target_player",
  "alliance_extension",
]);
export const NEEDS_TILE = new Set<ActionName>([
  "boat",
  "build",
  "launch_nuke",
  "spawn",
  "upgrade_structure",
  "move_warship",
  "cancel_boat",
  "delete_unit",
]);
export const NEEDS_QUANTITY = new Set<ActionName>([
  "attack",
  "expand",
  "boat",
  "donate_gold",
  "donate_troops",
]);

// Engine unit-type strings (matches rust/ofcore/src/feat.rs unit_class order/index).
export const UNIT_CLASSES = [
  "City",
  "Port",
  "Defense Post",
  "Missile Silo",
  "SAM Launcher",
  "Factory",
  "Warship",
  "Transport",
  "Trade Ship",
  "Atom Bomb",
  "Hydrogen Bomb",
  "MIRV",
  "SAMMissile",
  "MIRV Warhead",
  "Train",
] as const;
export const UNIT_CLASS_INDEX: Record<string, number> = Object.fromEntries(
  UNIT_CLASSES.map((n, i) => [n, i]),
);
export const STATIC_CLASSES = [
  "City",
  "Port",
  "Defense Post",
  "Missile Silo",
  "SAM Launcher",
  "Factory",
] as const;
export const NUM_STATIC = STATIC_CLASSES.length;
export const STATIC_INDICES = STATIC_CLASSES.map((n) => UNIT_CLASS_INDEX[n]);

export function logNorm(x: number): number {
  return Math.log10(1.0 + Math.max(0.0, x)) / 8.0;
}
