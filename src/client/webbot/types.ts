/** Loose shapes matching openfront/src/client/webbot/obsCore.ts's plain-object
 * output (entities()/legality()) - kept minimal/untyped-ish on purpose since
 * these mirror a JSON contract, not a class hierarchy. */

export interface EntityPlayer {
  id: number; // smallID
  pid: string; // persistent id (engine PlayerID)
  type: string;
  troops: number;
  gold: string;
  tiles: number;
  alive: boolean;
  traitor: boolean;
  embargoes: number[];
  reqsIn: number[];
  reqsOut: number[];
}

export interface EntityUnit {
  uid: number;
  type: string;
  owner: number;
  x: number;
  y: number;
  tx: number | null;
  ty: number | null;
  samLock: boolean;
  level: number;
  constructing: boolean;
  troops: number;
}

export interface EntityAttack {
  aid: number;
  from: number;
  to: number;
  troops: number;
  retreating: boolean;
}

export interface Entities {
  players: EntityPlayer[];
  alliances: [number, number, number][];
  units: EntityUnit[];
  attacks: EntityAttack[];
}

export interface LegalActions {
  attackable: number[];
  allianceRequestable: number[];
  allianceRejectable: number[];
  breakable: number[];
  targetable: number[];
  donatableGold: number[];
  donatableTroops: number[];
  embargoable: number[];
  buildableTypes: string[];
  canBoat: boolean;
  canExpand: boolean;
  hasSilo: boolean;
  troops: number;
  gold: string;
  attacks: number[];
  boats: number[];
  warships: number[];
  upgradable: number[];
  deletable: number[];
  stopEmbargoable: number[];
  extendable: number[];
}

export interface Legal {
  spawn: boolean;
  actions: Partial<LegalActions>;
}

export interface Choice {
  action: number;
  playerSlot?: number;
  tileRegion?: number;
  buildType?: number;
  nukeType?: number;
  quantityFrac?: number;
}
