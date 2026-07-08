/**
 * URL-triggered entry point: visiting the app with ?webbot=<gameID> skips
 * the normal UI and joins that lobby as a browser-native AI player (no
 * server GPU, no Python subprocess - see openfront/src/client/webbot/README
 * for the full pipeline). Wired from Main.ts's bootstrap.
 */
import { ACTIONS } from "./constants";
import { PlaySession } from "./PlaySession";
import type { WebBotDebugInfo } from "./bot";

interface RlDebugEntry {
  tick: number;
  desc: string;
  action: string;
  tiles: number;
  troops: number;
  value?: number;
  probs?: number[];
}

declare global {
  interface Window {
    __webbotDebug?: { actions: readonly string[]; log: RlDebugEntry[]; live: true };
    __webbotDone?: { winner: unknown; alive: boolean };
  }
}

const MAX_DEBUG_LOG = 500;

/** Feeds openfront/patches/client-replay-tooling.patch's RlDebugOverlay via
 * scripts/webbot_launcher.py's local /debug/<gameID> sidecar - same wire
 * format the old server-side rl.play --debug-port used to serve, just
 * sourced from this in-page decision loop instead of a Python process. */
function initDebugChannel(): (info: WebBotDebugInfo) => void {
  window.__webbotDebug = { actions: ACTIONS, log: [], live: true };
  return (info: WebBotDebugInfo) => {
    const log = window.__webbotDebug!.log;
    log.push({
      tick: info.tick,
      desc: `${info.action} (v${info.value >= 0 ? "+" : ""}${info.value.toFixed(2)})`,
      action: info.action,
      tiles: info.tiles,
      troops: info.troops,
      value: info.value,
      probs: info.probs,
    });
    if (log.length > MAX_DEBUG_LOG) log.shift();
  };
}

// Auth.ts's anonymous identity (player_persistent_id) lives in localStorage,
// which is shared by every tab of the same origin+profile. Left alone, a
// webbot tab run alongside a normal game tab in the same browser would look
// like the same player reconnecting twice, and the server would bounce the
// two sockets off each other in a reconnect loop. Route just that one key
// through sessionStorage (tab-scoped) so the bot always gets its own stable
// identity, independent of whatever else is open in this browser.
function isolateBotIdentity(): void {
  const KEY = "player_persistent_id";
  const original = window.localStorage;
  const proxy: Storage = new Proxy(original, {
    get(target, prop, receiver) {
      if (prop === "getItem") {
        return (key: string) =>
          key === KEY
            ? window.sessionStorage.getItem(KEY)
            : target.getItem(key);
      }
      if (prop === "setItem") {
        return (key: string, value: string) =>
          key === KEY
            ? window.sessionStorage.setItem(KEY, value)
            : target.setItem(key, value);
      }
      if (prop === "removeItem") {
        return (key: string) =>
          key === KEY
            ? window.sessionStorage.removeItem(KEY)
            : target.removeItem(key);
      }
      const value = Reflect.get(target, prop, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  Object.defineProperty(window, "localStorage", { value: proxy });
}

function mountStatusOverlay(): {
  setStatus: (s: string) => void;
  appendLog: (s: string) => void;
} {
  const el = document.createElement("div");
  el.id = "webbot-overlay";
  el.style.cssText = [
    "position:fixed",
    "top:8px",
    "left:8px",
    "z-index:99999",
    "background:rgba(0,0,0,0.85)",
    "color:#0f0",
    "font:12px/1.4 monospace",
    "padding:8px 12px",
    "border-radius:6px",
    "max-width:480px",
    "max-height:60vh",
    "overflow:auto",
    "white-space:pre-wrap",
  ].join(";");
  const status = document.createElement("div");
  status.style.cssText = "color:#fff;font-weight:bold;margin-bottom:4px;";
  status.textContent = "webbot: starting";
  const log = document.createElement("div");
  el.appendChild(status);
  el.appendChild(log);
  document.body.appendChild(el);
  return {
    setStatus: (s: string) => {
      status.textContent = `webbot: ${s}`;
    },
    appendLog: (s: string) => {
      const line = document.createElement("div");
      line.textContent = s;
      log.appendChild(line);
      while (log.childNodes.length > 200) log.removeChild(log.firstChild!);
      el.scrollTop = el.scrollHeight;
    },
  };
}

export async function startWebBot(params: URLSearchParams): Promise<void> {
  const gameID = params.get("webbot");
  if (!gameID) return;
  isolateBotIdentity();
  const { setStatus, appendLog } = mountStatusOverlay();
  const log = (msg: string) => {
    console.log(`[webbot] ${msg}`);
    appendLog(msg);
  };
  try {
    const onDecision = initDebugChannel();
    const session = new PlaySession({
      gameID,
      username: params.get("name") ?? undefined,
      greedy: params.get("greedy") === "1",
      log,
      onStatus: setStatus,
      onDecision,
      onGameEnded: (result) => {
        window.__webbotDone = result;
      },
    });
    await session.run();
  } catch (err) {
    setStatus("error");
    log(String(err));
    throw err;
  }
}
