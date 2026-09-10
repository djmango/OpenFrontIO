/**
 * Browser port of the WS join/turn-sync half of ClientGameRunner.ts: joins a
 * real OpenFront lobby using the same Transport/EventBus the human client
 * uses (so reconnect, ping, and the join -> start -> rejoin(0) -> start
 * handshake are all identical to production), mirrors the game locally via
 * the same core GameRunner the client's worker uses, and plays intents
 * chosen by WebBot - entirely client-side, no Node process, no server GPU.
 */
import { EventBus } from "../../core/EventBus";
import { createGameRunner, GameRunner } from "../../core/GameRunner";
import type { ErrorUpdate, GameUpdateViewData } from "../../core/game/GameUpdates";
import { GameUpdateType } from "../../core/game/GameUpdates";
import type { Intent, ServerMessage, Turn } from "../../core/Schemas";
import { LobbyConfig } from "../ClientGameRunner";
import { terrainMapFileLoader } from "../TerrainMapFileLoader";
import { Transport } from "../Transport";
import { WebBot, WebBotDebugInfo } from "./bot";

const DECISION_TICKS = 10;

export interface PlaySessionOptions {
  gameID: string;
  username?: string;
  greedy?: boolean;
  log?: (msg: string) => void;
  onStatus?: (status: string) => void;
  /** Fired after every WebBot.decide() - drives the MODEL debug overlay. */
  onDecision?: (info: WebBotDebugInfo) => void;
  /** Fired once when the game ends (win or elimination). */
  onGameEnded?: (result: { winner: unknown; alive: boolean }) => void;
}

export class PlaySession {
  private runner?: GameRunner;
  private myClientID = "";
  private turnsSeen = 0;
  private pendingTurns: Turn[] = [];
  private processing = false;
  private ended = false;
  private lastWinner: unknown = null;
  private transport: Transport;
  private bot: WebBot;
  private log: (msg: string) => void;
  private onStatus?: (status: string) => void;
  /** Once we've sent a spawn intent, don't send another until hasSpawned
   * (SpawnExecution re-rolls / can wipe a successful place). */
  private spawnSent = false;
  private spawnSentAtTick = -1;

  constructor(private opts: PlaySessionOptions) {
    this.log = opts.log ?? ((m) => console.log(`[webbot] ${m}`));
    this.onStatus = opts.onStatus;
    this.bot = new WebBot({ greedy: opts.greedy, onDebug: opts.onDecision });

    const lobbyConfig: LobbyConfig = {
      cosmetics: {},
      playerName: opts.username ?? "WebBot",
      playerClanTag: null,
      playerRole: null,
      gameID: opts.gameID,
      turnstileToken: null,
    };
    this.transport = new Transport(lobbyConfig, new EventBus());
  }

  async run(): Promise<void> {
    this.onStatus?.("loading models");
    await this.bot.load();

    this.onStatus?.("connecting");
    this.log(`connecting to game ${this.opts.gameID}`);
    this.transport.connect(
      () => {
        this.log("connected; joining");
        void this.transport.joinGame();
      },
      (msg) => void this.onServerMessage(msg),
    );
  }

  private async onServerMessage(msg: ServerMessage): Promise<void> {
    if (msg.type === "lobby_info") {
      this.myClientID = msg.myClientID;
      const startsAt = msg.lobby.startsAt;
      if (startsAt) {
        const secs = Math.max(0, Math.ceil((startsAt - Date.now()) / 1000));
        this.log(`lobby joined as ${this.myClientID}; starting in ~${secs}s`);
        this.onStatus?.(`starting in ~${secs}s`);
      } else {
        this.log(`lobby joined as ${this.myClientID}; waiting for start`);
        this.onStatus?.("waiting for start");
      }
      return;
    }
    if (msg.type === "error") {
      this.log(`server error: ${msg.error}${msg.message ? ` (${msg.message})` : ""}`);
      this.onStatus?.(`error: ${msg.error}`);
      return;
    }
    if (msg.type === "prestart") {
      this.log(`prestart: map ${msg.gameMap}`);
      this.onStatus?.("prestart");
      return;
    }
    if (msg.type === "desync") {
      this.log(`SERVER REPORTS DESYNC at turn ${msg.turn}`);
      return;
    }
    if (msg.type === "start" && this.runner === undefined) {
      this.myClientID = msg.myClientID ?? this.myClientID;
      this.log(
        `game starting: map ${msg.gameStartInfo.config.gameMap}, ` +
          `${msg.gameStartInfo.players.length} humans, me=${this.myClientID}`,
      );
      this.onStatus?.("initializing game");
      this.runner = await createGameRunner(
        msg.gameStartInfo,
        this.myClientID,
        terrainMapFileLoader,
        (gu) => this.onGameUpdate(gu),
      );
      this.bot.startGame(this.runner.game);
      this.onStatus?.("playing");
      // Spawn ASAP (synchronously - no ONNX) while the server is still in
      // spawn phase, THEN rejoin. An async/ONNX-based spawn pick used to
      // race rejoin(0)'s catch-up and land after the spawn window closed.
      this.sendImmediateSpawn();
      // Mirrors ClientGameRunner.start(): the initial "start" message only
      // carries config, not the turn log - rejoin(0) is what makes the
      // server actually (re)send turns from tick 0.
      void this.transport.rejoinGame(0);
      return;
    }
    if (msg.type === "start") {
      for (const turn of msg.turns) this.acceptTurn(turn);
      return;
    }
    if (msg.type === "turn") {
      this.acceptTurn(msg.turn);
    }
  }

  /** Fires the instant (ONNX-free) spawn pick right after the game is known
   * locally, well before rejoin(0) starts replaying turns. This is a normal
   * client spawn intent - identical in kind to what a human's click sends -
   * so it needs no server-side config changes and doesn't touch how real
   * human players spawn. */
  private sendImmediateSpawn(): void {
    if (this.spawnSent || this.runner === undefined) return;
    try {
      const spawn = this.bot.instantSpawn();
      if (spawn === null) {
        this.log("immediate spawn: no valid tile (map full?)");
        return;
      }
      this.spawnSent = true;
      this.spawnSentAtTick = this.runner.game.ticks();
      this.log(`immediate spawn send tile=${(spawn as { tile?: number }).tile}`);
      this.transport.sendRawIntent(spawn as unknown as Intent);
    } catch (err) {
      this.log(`immediate spawn failed: ${String(err)}`);
    }
  }

  private acceptTurn(turn: Turn): void {
    if (turn.turnNumber < this.turnsSeen) return;
    while (turn.turnNumber - 1 > this.turnsSeen) {
      this.enqueue({ turnNumber: this.turnsSeen, intents: [] });
    }
    this.enqueue(turn);
  }

  private enqueue(turn: Turn): void {
    this.turnsSeen++;
    this.pendingTurns.push(turn);
    void this.drain();
  }

  private onGameUpdate(gu: GameUpdateViewData | ErrorUpdate): void {
    if ("errMsg" in gu) {
      this.log(`engine error: ${gu.errMsg}`);
      this.ended = true;
      return;
    }
    const winUpdates = gu.updates[GameUpdateType.Win];
    if (winUpdates && winUpdates.length > 0) {
      this.lastWinner = winUpdates[0].winner;
    }
  }

  /** Process turns strictly in order. Only send intents when caught up to
   * the live tip — during rejoin(0) catch-up, deciding/sending would fire
   * spawn intents that arrive after the server's spawn window already closed. */
  private async drain(): Promise<void> {
    if (this.processing || this.ended || this.runner === undefined) return;
    this.processing = true;
    const runner = this.runner;
    const game = runner.game;
    while (this.pendingTurns.length > 0) {
      const turn = this.pendingTurns.shift()!;
      runner.addTurn(turn);
      runner.executeNextTick();

      const me = game.playerByClientID(this.myClientID);
      const dead = !game.inSpawnPhase() && me !== null && !me.isAlive();
      if (this.lastWinner !== null || dead) {
        this.log(`game ended: winner=${JSON.stringify(this.lastWinner)} alive=${!dead}`);
        this.onStatus?.(dead ? "eliminated" : "game over");
        this.ended = true;
        this.opts.onGameEnded?.({ winner: this.lastWinner, alive: !dead });
        break;
      }

      // Still replaying history — don't send intents against a stale tick.
      if (this.pendingTurns.length > 0) continue;

      if (game.ticks() % DECISION_TICKS !== 0) continue;

      try {
        const meNow = game.playerByClientID(this.myClientID);
        const spawned = meNow?.hasSpawned() ?? false;
        if (!spawned) {
          if (this.spawnSent && game.ticks() - this.spawnSentAtTick < 20) {
            continue; // give the last spawn intent time to land
          }
          if (this.spawnSent) {
            this.log(`t=${game.ticks()} spawn not landed, retrying`);
            this.spawnSent = false;
          }
          // Retry via the fast path too - don't rely on the policy's own
          // action sampling to happen to pick "spawn" again.
          this.sendImmediateSpawn();
          continue;
        }
        this.spawnSent = false;
        const intents = await this.bot.decide(game, this.myClientID);
        // ONNX is slower than the turn clock, so pendingTurns is almost
        // always non-empty after decide(). Still send — the pre-decide
        // `pendingTurns.length > 0` continue already skips history replay.
        if (intents.some((i) => i.type === "spawn")) {
          this.spawnSent = true;
          this.spawnSentAtTick = game.ticks();
        }
        if (intents.length > 0) {
          this.log(
            `t=${game.ticks()} send ${intents.map((i) => i.type).join(",")} ` +
              `tiles=${meNow?.numTilesOwned() ?? 0} spawned=${spawned}`,
          );
        }
        for (const intent of intents) {
          this.transport.sendRawIntent(intent as unknown as Intent);
        }
      } catch (err) {
        this.log(`decide() failed: ${String(err)}`);
      }
    }
    this.processing = false;
    // Turns arrived during decide(); drain again.
    if (!this.ended && this.pendingTurns.length > 0) void this.drain();
  }
}
