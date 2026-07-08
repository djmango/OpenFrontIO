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
import { WebBot } from "./bot";

const DECISION_TICKS = 10;

export interface PlaySessionOptions {
  gameID: string;
  username?: string;
  greedy?: boolean;
  log?: (msg: string) => void;
  onStatus?: (status: string) => void;
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

  constructor(private opts: PlaySessionOptions) {
    this.log = opts.log ?? ((m) => console.log(`[webbot] ${m}`));
    this.onStatus = opts.onStatus;
    this.bot = new WebBot({ greedy: opts.greedy });

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
      this.log(`lobby joined as ${this.myClientID}; waiting for host to start`);
      this.onStatus?.("waiting for host");
      return;
    }
    if (msg.type === "error") {
      this.log(`server error: ${msg.error}${msg.message ? ` (${msg.message})` : ""}`);
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

  /** Process turns strictly in order; pause on each decision tick to await
   * the (async, ONNX-backed) WebBot before sending intents. */
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
        break;
      }

      if (game.ticks() % DECISION_TICKS === 0) {
        try {
          const intents = await this.bot.decide(game, this.myClientID);
          for (const intent of intents) {
            this.transport.sendRawIntent(intent as unknown as Intent);
          }
        } catch (err) {
          this.log(`decide() failed: ${String(err)}`);
        }
      }
    }
    this.processing = false;
  }
}
