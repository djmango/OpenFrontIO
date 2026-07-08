/**
 * Engine-state -> plain-object observation featurization, shared by the
 * Node-side RL bridge (bridge/common.ts) and the in-browser WebBot
 * (openfront/src/client/webbot/*). Lives under src/client so it bundles
 * cleanly into the browser app; bridge/common.ts re-exports it for the
 * training pipeline. Zero Node dependencies (no Buffer/zlib) so it works
 * unmodified in both environments.
 */
import { Game, Player, UnitType } from "../../core/game/Game";

export const STRUCTURES = [
  UnitType.City,
  UnitType.Port,
  UnitType.DefensePost,
  UnitType.MissileSilo,
  UnitType.SAMLauncher,
  UnitType.Factory,
];
export const LAUNCHABLE = [
  UnitType.AtomBomb,
  UnitType.HydrogenBomb,
  UnitType.MIRV,
];

export function entities(game: Game): object {
  const config = game.config();
  // allPlayers(): game.players() filters to living players, which would
  // drop the dead agent (and everyone eliminated) from placement math.
  const players = game.allPlayers().map((p) => ({
    id: p.smallID(),
    pid: p.id(), // persistent ID; intents reference players by this
    type: p.type(),
    troops: Math.round(p.troops()),
    gold: p.gold().toString(),
    tiles: p.numTilesOwned(),
    alive: p.isAlive(),
    traitor: p.isTraitor(),
    embargoes: p.getEmbargoes().map((e) => e.target.smallID()),
    reqsIn: p.incomingAllianceRequests().map((r) => r.requestor().smallID()),
    reqsOut: p.outgoingAllianceRequests().map((r) => r.recipient().smallID()),
    // Targeting marks: which players this one has painted as a priority
    // target for allies (v7).
    targets: p.targets().map((t) => t.smallID()),
    // Growth rates (v7): deterministic given current state, cheap for every
    // player, not just the agent - lets the policy compare trajectories.
    troopIncome: p.isAlive() ? config.troopIncreaseRate(p) : 0,
    goldIncome: p.isAlive() ? config.goldAdditionRate(p).toString() : "0",
    // Doomsday Clock (anti-stall): marked + ticks spent below the bar (v7).
    doomsday: p.inDoomsdayClock(),
    doomsdayTicks: p.doomsdayClockTicks(),
  }));

  const alliances: number[][] = [];
  const seen = new Set<string>();
  for (const p of game.players()) {
    for (const a of p.alliances()) {
      const x = a.requestor().smallID();
      const y = a.recipient().smallID();
      const key = x < y ? `${x}:${y}` : `${y}:${x}`;
      if (!seen.has(key)) {
        seen.add(key);
        alliances.push([x, y, a.expiresAt()]);
      }
    }
  }

  const units = game.players().flatMap((p) =>
    p.units().map((u) => {
      const tt = u.targetTile();
      return {
        uid: u.id(),
        type: u.type(),
        owner: p.smallID(),
        x: game.x(u.tile()),
        y: game.y(u.tile()),
        tx: tt !== undefined ? game.x(tt) : null,
        ty: tt !== undefined ? game.y(tt) : null,
        samLock: u.targetedBySAM(),
        level: u.level(),
        health: u.hasHealth() ? u.health() : null,
        maxHealth: u.hasHealth() ? u.maxHealth() : null,
        constructing: u.isUnderConstruction(),
        cooldown: u.isInCooldown(),
        station: u.hasTrainStation(),
        troops: Math.round(u.troops()),
      };
    }),
  );

  const attacks = game.players().flatMap((p) =>
    p.outgoingAttacks().map((a) => {
      const src = a.sourceTile();
      return {
        aid: a.id(),
        from: p.smallID(),
        to: a.target().isPlayer() ? a.target().smallID() : 0,
        troops: Math.round(a.troops()),
        retreating: a.retreating(),
        srcX: src !== null ? game.x(src) : null,
        srcY: src !== null ? game.y(src) : null,
      };
    }),
  );

  return {
    players,
    alliances,
    units,
    attacks,
    doomsdayEnabled: config.doomsdayClockConfig().enabled,
  };
}

/** Any of the player's border tiles touches water (a boat could launch). */
export function hasShoreBorder(game: Game, player: Player): boolean {
  for (const t of player.borderTiles()) {
    if (game.isShore(t)) return true;
  }
  return false;
}

/** Any of the player's border tiles touches conquerable neutral land
 * (an expand attack would actually take tiles instead of fizzling). */
export function bordersNeutralLand(game: Game, player: Player): boolean {
  let found = false;
  for (const t of player.borderTiles()) {
    game.forEachNeighbor(t, (n) => {
      // Game has no isImpassable(); the Node bridge only "worked" because it
      // called a method that doesn't exist on this interface (datagen/replay.ts
      // shims it to always return false) - so this check was always a no-op.
      // Preserve that observed (if accidental) behavior here.
      if (game.isLand(n) && !game.hasOwner(n)) {
        found = true;
      }
    });
    if (found) return true;
  }
  return false;
}

/** Exact per-action legality from engine calls; the policy builds masks. */
export function legality(game: Game, clientID: string): object {
  const agent = game.playerByClientID(clientID) ?? null;
  if (!agent || !agent.isAlive()) {
    return { spawn: game.inSpawnPhase(), actions: {} };
  }
  const others = game.players().filter((p) => p !== agent && p.isAlive());
  const gold = agent.gold();
  const buildable = [...STRUCTURES, ...LAUNCHABLE, UnitType.Warship].filter(
    (t) => gold >= game.unitInfo(t).cost(game, agent),
  );
  return {
    spawn: game.inSpawnPhase(),
    actions: {
      attackable: others
        .filter((p) => agent.sharesBorderWith(p) && !agent.isFriendly(p))
        .map((p) => p.smallID()),
      allianceRequestable: others
        .filter((p) => agent.canSendAllianceRequest(p))
        .map((p) => p.smallID()),
      allianceRejectable: agent
        .incomingAllianceRequests()
        .map((r) => r.requestor().smallID()),
      breakable: agent.alliances().map((a) => a.other(agent).smallID()),
      targetable: others
        .filter((p) => agent.canTarget(p))
        .map((p) => p.smallID()),
      donatableGold: others
        .filter((p) => agent.canDonateGold(p))
        .map((p) => p.smallID()),
      donatableTroops: others
        .filter((p) => agent.canDonateTroops(p))
        .map((p) => p.smallID()),
      embargoable: others
        .filter((p) => !agent.hasEmbargoAgainst(p))
        .map((p) => p.smallID()),
      buildableTypes: buildable,
      // Mirrors TransportShipExecution.init(): a boat needs a free slot
      // under boatMaxNumber and a shore to launch from. (Destination
      // validity is checked per-intent in env.ts step().)
      canBoat:
        agent.units(UnitType.TransportShip).length <
          game.config().boatMaxNumber() && hasShoreBorder(game, agent),
      // An expand (attack with null target) only conquers when neutral
      // land actually borders the player; otherwise it fizzles.
      canExpand: bordersNeutralLand(game, agent),
      // Mirrors nukeSpawn(): a silo on cooldown (or spawn immunity) makes
      // every launch a silent no-op.
      hasSilo:
        !game.isSpawnImmunityActive() &&
        agent
          .units(UnitType.MissileSilo)
          .some(
            (u) =>
              !u.isUnderConstruction() && u.isActive() && !u.isInCooldown(),
          ),
      troops: Math.round(agent.troops()),
      gold: gold.toString(),
      attacks: agent.outgoingAttacks().map((a) => a.id()),
      boats: agent.units(UnitType.TransportShip).map((u) => u.id()),
      warships: agent
        .units(UnitType.Warship)
        .filter((u) => u.isActive())
        .map((u) => u.id()),
      // Exact upgrade legality (type upgradable + not constructing + owned)
      // plus affordability - the upgrade costs the same as building anew.
      upgradable: agent
        .units()
        .filter(
          (u) =>
            agent.canUpgradeUnit(u) &&
            gold >= game.unitInfo(u.type()).cost(game, agent),
        )
        .map((u) => u.id()),
      // Mirrors DeleteUnitExecution: active unit standing on the player's
      // own land territory, delete cooldown expired.
      deletable: agent.canDeleteUnit()
        ? agent
            .units()
            .filter((u) => {
              if (!u.isActive() || !game.isLand(u.tile())) return false;
              const o = game.owner(u.tile());
              return o.isPlayer() && o.id() === agent.id();
            })
            .map((u) => u.id())
        : [],
      // Players the agent currently embargoes (embargo stop targets).
      stopEmbargoable: agent.getEmbargoes().map((e) => e.target.smallID()),
      // Alliances inside the extension window not yet agreed by the agent.
      extendable: agent
        .alliances()
        .map((a) => a.other(agent))
        .filter((p) => agent.allianceInfo(p)?.canExtend === true)
        .map((p) => p.smallID()),
    },
  };
}
