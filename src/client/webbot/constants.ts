/**
 * Constants ported 1:1 from the training pipeline (rl/obs.py, ae/units.py,
 * rl/curriculum.py). Keep these in sync with the Python source - they must
 * match exactly for the ONNX-exported AE encoder + policy to see the same
 * tensor layout they were trained on.
 */

export const IS_LAND_BIT = 7;
export const SHORELINE_BIT = 6;
export const MAGNITUDE_MASK = 0x1f;
export const MAGNITUDE_NORM = 31.0;
export const IMPASSABLE_MAGNITUDE = 31;

// Tile-state (GameMap.tileStateBuffer) bit layout, see TileCodec.ts.
export const OWNER_MASK = 0xfff;
export const FALLOUT_BIT = 1 << 13;

// v4 spatial resolution: one latent cell / tile-pointer region per 8x8 tiles.
export const REGION = 8;
export const LATENT_C = 32;
export const MAX_SLOTS = 128;

export const N_TRANSIENT = 8;
export const C_GRID = LATENT_C + 3 + N_TRANSIENT;

export const LOCAL = 64;
export const N_LOCAL = 4;

export const P_FEAT = 12;
export const N_SCALARS = 8;

// Largest featurized grid across the curriculum (Asia at the v4 1/8 latent
// resolution) - region indices use this stride regardless of the actual
// map's grid size (see IntentTranslator.region_tile in rl/ppo_translate.py).
export const GH_MAX = 150;
export const GW_MAX = 250;

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

// Engine unit-type strings (matches ae/units.py UNIT_CLASSES order/index).
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
