import {
  lazy,
  memo,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
  useMemo,
} from "react";
import {
  advance,
  applyShot,
  bearingTo,
  CLUBS,
  distance,
  heightAt,
  newRound,
  restoreRound,
  scoreName,
  simulate,
  surfaceAt,
  type Course,
  type PracticeHole,
  type Round,
  type ShotResult,
  type Tile,
  type Vec3,
  type World,
} from "./engine";
import { COURSE_ROOT, loadTile } from "./terrain";
import type { Controls, TravelMode } from "./CourseScene";
import "./course.css";
import CourseMap,{HoleLinks} from "./CourseMap";
import {DriveStick} from "./DriveStick";
import { golfSound } from "./audio";
import {useSwing} from "./useSwing";
import SwingPanel from "./SwingPanel";
import type {Strike,SwingMode} from "./swing";
import {ErrorBoundary} from '../components/shared/ErrorBoundary';

const Scene = lazy(() => import("./CourseScene"));
const SAVE = "strikelab.grenland.round.v1";
const PREFS = "strikelab.grenland.conditions.v1";
type LaunchEvent = {
  type: string;
  version: number;
  seq: number;
  shot_id: string;
  device_id: string;
  speed_m_s: number;
  direction_deg: number;
  launch_angle_deg: number | null;
  spin_rpm: number | null;
  direction_frame: string;
  quality: { speed: string; direction: string; launch: string; spin: string };
};

function openingAim(course:Course, round:Round){
  const hole=course.practice[round.hole],waypoint=hole.route?.[1];
  if(!waypoint||round.strokes!==0||distance(round.ball,hole.tee)>2)return 0;
  const delta=bearingTo(round.ball,[waypoint[0],round.ball[1],waypoint[1]])-bearingTo(round.ball,hole.pin);
  return Math.atan2(Math.sin(delta),Math.cos(delta))*180/Math.PI;
}

const MiniMap=memo(function MiniMap({ course, round }: { course: Course; round: Round }) {
  const hole = course.practice[round.hole],
    center: Vec3 = [
      (round.ball[0] + hole.pin[0]) / 2,
      0,
      (round.ball[2] + hole.pin[2]) / 2,
    ];
  const span = Math.max(120, distance(round.ball, hole.pin) * 1.4);
  return (
    <svg
      className="course-map"
      viewBox={`${center[0] - span / 2} ${center[2] - span / 2} ${span} ${span}`}
      role="img"
      aria-label="Green map with your ball and target flag"
    >
      <rect
        x={center[0] - span}
        y={center[2] - span}
        width={span * 2}
        height={span * 2}
        fill="#233e2b"
      />
      {course.regions
        .filter((r) => ["green", "fairway", "bunker", "trees"].includes(r.kind))
        .map((r) => (
          <polygon
            key={r.id}
            points={r.points.map((p) => p.join(",")).join(" ")}
            fill={
              {
                green: "#92b26b",
                fairway: "#55754a",
                bunker: "#c5b88d",
                trees: "#193522",
              }[r.kind]
            }
          />
        ))}
      {hole.route && (
        <polyline
          points={hole.route.map((p) => p.join(",")).join(" ")}
          stroke="#ddba78"
          fill="none"
          strokeWidth={span / 180}
          strokeDasharray={`${span / 70} ${span / 70}`}
        />
      )}
      <line
        x1={round.ball[0]}
        y1={round.ball[2]}
        x2={hole.pin[0]}
        y2={hole.pin[2]}
        stroke="#eee4bb"
        strokeWidth={span / 250}
        strokeDasharray={`${span / 60} ${span / 60}`}
      />
      <circle cx={hole.pin[0]} cy={hole.pin[2]} r={span / 50} fill="#e18a56" />
      <circle
        cx={round.ball[0]}
        cy={round.ball[2]}
        r={span / 65}
        fill="white"
      />
    </svg>
  );
});

function Game({
  course,
  terrain,
  onCourseMode,
  exploreHole,
}: {
  course: Course;
  exploreHole?:number;
  terrain: Tile;
  onCourseMode: (mode: "preview" | "practice") => void;
}) {
  const saveKey = exploreHole ? SAVE + ".explore." + exploreHole : course.mode === "preview" ? SAVE + ".preview" : SAVE;
  const [round, setRound] = useState<Round>(() => {
    try {
      return (
        restoreRound(localStorage.getItem(saveKey), course) ?? newRound(course)
      );
    } catch {
      return newRound(course);
    }
  });
  const roundRef = useRef(round);
  const [detail, setDetail] = useState<Tile>();
  const [error, setError] = useState("");
  const [sceneReady,setSceneReady]=useState(false);
  const sceneLoaded=useCallback(()=>setSceneReady(true),[]);
  const [club, setClub] = useState(()=>distance(round.ball,course.practice[round.hole].pin)<35?13:course.mode === "preview" ? 0 : 9),
    [power, setPower] = useState(()=>{const d=distance(round.ball,course.practice[round.hole].pin);return d<35?Math.max(1,Math.min(100,Math.round(Math.sqrt(2*9.81*.056*d)/CLUBS[13].speed*100))):90;}),
    [aim, setAim] = useState(()=>openingAim(course,round));
  const [stimp, setStimp] = useState(() => {
    try {
      const n = Number(JSON.parse(localStorage.getItem(PREFS) ?? "{}").stimp);
      return n >= 6 && n <= 14 ? n : 10;
    } catch {
      return 10;
    }
  });
  const [wind, setWind] = useState(0);
  const [sound, setSound] = useState(() => {
    try {
      return localStorage.getItem("strikelab.golf.sound") !== "off";
    } catch {
      return true;
    }
  });
  const soundRef = useRef(sound);
  soundRef.current = sound;
  const [mode, setMode] = useState<TravelMode>("golf"),
    [quality, setQuality] = useState<"balanced" | "high">("balanced");
  const [showMap,setShowMap]=useState(false),[boost,setBoost]=useState(false);
  const boostRef=useRef(boost);useEffect(()=>{boostRef.current=boost;},[boost]);
  const closeMap=useCallback(()=>setShowMap(false),[]);
  const [showScore, setShowScore] = useState(false),
    [showHelp, setShowHelp] = useState(false),
    [showSettings, setShowSettings] = useState(false);
  const [view, setView] = useState<"practice" | "official">("practice");
  const [motion, setMotion] = useState(0),
    [animation, setAnimation] = useState<ShotResult>(),
    [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [simEnabled, setSimEnabled] = useState(false),
    [simStatus, setSimStatus] = useState("Not connected");
  const [simUrl, setSimUrl] = useState(() => {
    try {
      return (
        localStorage.getItem("strikelab.sim.url") ||
        `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.port === "8088" ? `${location.hostname}:8000` : location.host}/ws/shots`
      );
    } catch {
      return `ws://127.0.0.1:8000/ws/shots`;
    }
  });
  const [pad, setPad] = useState(false);
  const [swingMode,setSwingMode]=useState<SwingMode>("analog");
  const [cameraMode,setCameraMode]=useState<"player"|"scout">("player");
  const [showGrid,setShowGrid]=useState(true);
  const [journey,setJourney]=useState<Vec3>();
  const [travelDistance,setTravelDistance]=useState(0);
  const [strikeFeedback,setStrikeFeedback]=useState("");
  const [caddieBusy, setCaddieBusy] = useState(false),
    [caddieNote, setCaddieNote] = useState("");
  const worker = useRef<Worker | null>(null);
  useEffect(() => () => worker.current?.terminate(), []);
  useEffect(() => {
    worker.current?.terminate();
    setCaddieBusy(false);
    setCaddieNote("");
  }, [club, stimp, wind, round.hole, round.ball]);
  const controls = useRef<Controls>({ forward: 0, turn: 0, gamepad: false });
  const touch = useRef({ forward: 0, turn: 0 });
  const hole = course.practice[round.hole];
  const world: World = useMemo(
    () => ({ terrain, detail, pin: hole.pin, stimp, wind: [wind, 0] }),
    [terrain, detail, hole.pin, stimp, wind],
  );
  const latest = useRef({
    world,
    aim,
    club,
    power,
    mode,
    detailReady: false,
    modal: false,
  });
  const detailReady = detail?.url === hole.tile.url && sceneReady;
  useEffect(() => {
    latest.current = {
      world,
      aim,
      club,
      power,
      mode,
      detailReady,
      modal: showScore || showHelp || showSettings || showMap,
    };
  });
  const persist = useCallback(
    (next: Round) => {
      // Save outcome, score, cursor and event IDs together before acknowledging a shot.
      try {
        localStorage.setItem(saveKey, JSON.stringify(next));
      } catch {
        setError(
          "Round could not be saved. Free some browser storage before continuing.",
        );
        return false;
      }
      roundRef.current = next;
      setRound(next);
      return true;
    },
    [saveKey],
  );
  useEffect(() => {
    const controller = new AbortController();
    loadTile(hole.tile, controller.signal)
      .then(setDetail)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => controller.abort();
  }, [hole]);
  const takeShot = useCallback(
    (event?: LaunchEvent, strike?:Strike) => {
      const r = roundRef.current,
        l = latest.current;
      if (
        busyRef.current ||
        l.modal ||
        !l.detailReady ||
        l.mode !== "golf" ||
        r.complete ||
        r.scores[r.hole] !== null
      )
        return false;
      if (event && r.accepted.includes(event.shot_id)) return true;
      const relative = event?.direction_deg ?? l.aim+(strike?.face??0);
      const shot = {
        speed: event?.speed_m_s ?? (CLUBS[l.club].speed * l.power) / 100*(strike?.power??1)*(strike?.contact??1),
        bearing: bearingTo(r.ball, l.world.pin) + (relative * Math.PI) / 180,
        launch: event?.launch_angle_deg ?? (event ? 0 : CLUBS[l.club].loft),
        spin: event?.spin_rpm ?? (event ? 0 : CLUBS[l.club].spin),
      };
      try {
        const result = simulate(l.world, r.ball, shot);
        const next = applyShot(r, result, event?.shot_id);
        if (
          !persist(
            event
              ? { ...next, cursor: event.seq, device: event.device_id }
              : next,
          )
        )
          return false;
        busyRef.current = true;
        setBusy(true);
        setCameraMode("player");
        setStrikeFeedback(strike?`${strike.label} · ${Math.round(strike.power*100)}% swing · ${Math.abs(strike.face).toFixed(1)}° ${strike.face<0?"left":"right"}`:"Simulator shot");
        setAnimation(result);
        setMotion((n) => n + 1);
        setAim(0);
        setError("");
        if (soundRef.current) golfSound("hit");
        return true;
      } catch (e) {
        setError(
          e instanceof Error ? e.message : "Shot could not be simulated",
        );
        return false;
      }
    },
    [persist],
  );
  const swing=useSwing((strike)=>takeShot(undefined,strike),detailReady&&!busy&&!round.complete&&round.scores[round.hole]===null&&mode==="golf"&&!showHelp&&!showScore&&!showSettings&&!showMap,swingMode);
  const swingAPI=useRef(swing);swingAPI.current=swing;
  const settled = useCallback(() => {
    swingAPI.current.reset();
    busyRef.current = false;
    setBusy(false);
    const r = roundRef.current,
      w = latest.current.world;
    if (r.last?.made) {
      if (soundRef.current) golfSound("cup");
      return;
    }
    if (surfaceAt(w, r.ball[0], r.ball[2]) === "green") {
      setClub(13);
      setPower(
        Math.max(
          1,
          Math.min(
            100,
            (Math.sqrt(
              ((2 * 9.81 * 0.56) / w.stimp) * distance(r.ball, w.pin),
            ) /
              CLUBS[13].speed) *
              100,
          ),
        ),
      );
    }
  }, []);
  const nextHole = useCallback(() => {
    if (busyRef.current||roundRef.current.scores[roundRef.current.hole]===null) return;
    const next = advance(roundRef.current, course);
    if (persist(next)) {
      swingAPI.current.reset();
      setCameraMode("player");
      setAnimation(undefined);
      setAim(openingAim(course,next));
      setPower(90);
      setClub(
        course.mode === "preview"
          ? distance(next.ball, course.practice[next.hole].pin) > 180
            ? 0
            : 6
          : 9,
      );
    }
  }, [course, persist]);
  useEffect(() => {
    const keys = new Set<string>();
    let raf = 0,
      last = performance.now();
    let oldButtons: boolean[] = [];
    let padSwing = false;
    const onDown = (event: KeyboardEvent) => {
      if (event.code === "Escape") {
        setShowScore(false);
        setShowSettings(false);
        setShowHelp(false);
        setShowMap(false);
        return;
      }
      if (latest.current.modal) return;
      if (
        (event.target as HTMLElement)?.closest("input,select,textarea,button")
      )
        return;
      if (
        ["Space", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(
          event.code,
        )
      )
        event.preventDefault();
      keys.add(event.code);
      if (event.repeat) return;
      if (event.code === "Space") swingAPI.current.press();
      if (event.code === "KeyV") setCameraMode(v=>v==="player"?"scout":"player");
      if (event.code === "KeyG") setShowGrid(v=>!v);
      if (event.code === "KeyN") nextHole();
      if (event.code === "KeyC"&&!busyRef.current) setClub((c) => (c + 1) % CLUBS.length);
      if (event.code === "Tab") {
        event.preventDefault();
        setShowScore((v) => !v);
      }
      if (event.code === "Escape") {
        setShowScore(false);
        setShowSettings(false);
        setShowHelp(false);
      }
    };
    const onUp = (event: KeyboardEvent) => keys.delete(event.code);
    const clear = () => {
      swingAPI.current.reset();
      padSwing = false;
      keys.clear();
      touch.current = { forward: 0, turn: 0 };
      controls.current.forward = 0;
      controls.current.turn = 0;
    };
    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const gamepad = Array.from(navigator.getGamepads?.() ?? []).find(
        (p) => p?.connected,
      );
      const dead = (n: number) => (Math.abs(n) > 0.16 ? n : 0);
      const buttons = gamepad?.buttons.map((b) => b.pressed) ?? [];
      if (!!gamepad !== controls.current.gamepad) {
        controls.current.gamepad = !!gamepad;
        setPad(!!gamepad);
      }
      const pressed = (i: number) => buttons[i] && !oldButtons[i];
      if (pressed(0)) swingAPI.current.press();
      const pull=gamepad?.axes[3]??0;
      if(gamepad&&!latest.current.modal&&latest.current.mode==="golf"){
        if(pull>.15&&swingAPI.current.controller.current.phase==="ready"){swingAPI.current.begin();padSwing=true;}
        if(padSwing&&["backswing","downswing"].includes(swingAPI.current.controller.current.phase))swingAPI.current.move(Math.max(0,pull),gamepad.axes[2]??0);
        else padSwing=false;
      }else if(padSwing){swingAPI.current.reset();padSwing=false;
      }
      if(pressed(2))setCameraMode(v=>v==="player"?"scout":"player");
      if (pressed(1)) {
        if (latest.current.modal) {
          setShowScore(false);
          setShowSettings(false);
          setShowHelp(false);
        } else nextHole();
      }
      if (pressed(4)&&!busyRef.current&&!latest.current.modal) setClub((c) => (c + CLUBS.length - 1) % CLUBS.length);
      if (pressed(5)&&!busyRef.current&&!latest.current.modal) setClub((c) => (c + 1) % CLUBS.length);
      if (pressed(3) && !busyRef.current && !latest.current.modal)
        setMode((m) =>
          m === "golf" ? "walk" : m === "walk" ? "cart" : "golf",
        );
      if (pressed(9)) setShowScore((v) => !v);
      oldButtons = buttons;
      const horizontal =
        Number(keys.has("ArrowRight")) -
        Number(keys.has("ArrowLeft")) +
        (buttons[15] ? 1 : 0) -
        (buttons[14] ? 1 : 0) +
        dead(gamepad?.axes[0] ?? 0);
      const vertical =
        Number(keys.has("ArrowUp")) -
        Number(keys.has("ArrowDown")) +
        (buttons[12] ? 1 : 0) -
        (buttons[13] ? 1 : 0) -
        dead(gamepad?.axes[1] ?? 0);
      if (latest.current.mode === "golf" && !latest.current.modal && !busyRef.current) {
        if (horizontal)
          setAim((a) =>
            Math.max(-180, Math.min(180, a + horizontal * dt * 14)),
          );
        if (vertical)
          setPower((p) => Math.max(1, Math.min(100, p + vertical * dt * 22)));
      }
      controls.current.forward =
        touch.current.forward +
        Number(keys.has("KeyW")) -
        Number(keys.has("KeyS")) +
        vertical;
      controls.current.turn =
        touch.current.turn +
        Number(keys.has("KeyD")) -
        Number(keys.has("KeyA")) +
        horizontal;
      controls.current.forward=Math.max(-1,Math.min(1,controls.current.forward));
      controls.current.turn=Math.max(-1,Math.min(1,controls.current.turn));
      controls.current.boost=boostRef.current||keys.has('ShiftLeft')||keys.has('ShiftRight')||!!buttons[7];
      if (latest.current.modal) {
        controls.current.boost=false;
        controls.current.forward = 0;
        controls.current.turn = 0;
      }
      raf = requestAnimationFrame(frame);
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    window.addEventListener("blur", clear);
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
      window.removeEventListener("blur", clear);
    };
  }, [takeShot, nextHole]);
  useEffect(() => {
    if (!simEnabled) return;
    let stopped = false,
      socket: WebSocket | undefined,
      retry: ReturnType<typeof setTimeout>;
    const pending: LaunchEvent[] = [];
    const connect = () => {
      if (stopped) return;
      try {
        const url = new URL(simUrl);
        if (!["ws:", "wss:"].includes(url.protocol))
          throw new Error("Use a ws:// or wss:// simulator address");
        socket = new WebSocket(url);
        setSimStatus("Connecting…");
      } catch (e) {
        setSimStatus(
          e instanceof Error ? e.message : "Invalid simulator address",
        );
        return;
      }
      socket.onopen = () => {
        const r = roundRef.current;
        socket?.send(
          JSON.stringify({
            type: "subscribe",
            client_id: r.id,
            ...(r.cursor !== undefined ? { after: r.cursor } : {}),
          }),
        );
      };
      socket.onmessage = (message) => {
        try {
          const event = JSON.parse(message.data);
          if (event.type === "subscribed") {
            const r = roundRef.current;
            if (
              !Number.isSafeInteger(event.cursor) ||
              event.cursor < 0 ||
              typeof event.device_id !== "string"
            )
              throw new Error("Invalid subscription");
            if (r.device && r.device !== event.device_id) {
              setSimStatus("Different simulator. Start a new session to pair.");
              socket?.close();
              stopped = true;
              return;
            }
            if (
              !persist({ ...r, cursor: event.cursor, device: event.device_id })
            ) {
              socket?.close();
              stopped = true;
              return;
            }
            setSimStatus("Connected · ready for a real putt");
          } else if (event.type === "shot.launched") {
            if (
              event.version !== 1 ||
              typeof event.shot_id !== "string" ||
              typeof event.device_id !== "string" ||
              !Number.isSafeInteger(event.seq) ||
              event.direction_frame !== "target-relative-positive-right" ||
              !Number.isFinite(event.speed_m_s) ||
              !Number.isFinite(event.direction_deg) ||
              (event.launch_angle_deg !== null &&
                !Number.isFinite(event.launch_angle_deg)) ||
              (event.spin_rpm !== null && !Number.isFinite(event.spin_rpm))
            )
              throw new Error("Invalid launch contract");
            if (
              event.device_id !== roundRef.current.device ||
              pending.length >= 32
            )
              throw new Error("Unexpected simulator event");
            pending.push(event);
          }
        } catch {
          setSimStatus("Invalid simulator event; connection closed");
          stopped = true;
          socket?.close();
        }
      };
      socket.onclose = () => {
        pending.length = 0;
        if (!stopped) {
          setSimStatus("Disconnected · retrying");
          retry = setTimeout(connect, 2000);
        }
      };
      socket.onerror = () => setSimStatus("Simulator unavailable");
    };
    connect();
    const heartbeat = setInterval(() => {
      if (socket?.readyState !== WebSocket.OPEN) return;
      const event = pending[0];
      if (!event) return;
      const r = roundRef.current;
      if (r.accepted.includes(event.shot_id) || event.seq <= (r.cursor ?? 0)) {
        socket.send(JSON.stringify({ type: "ack", seq: event.seq }));
        pending.shift();
        return;
      }
      if (
        !event.quality ||
        !["measured", "estimated"].includes(event.quality.speed) ||
        event.quality.direction !== "measured"
      ) {
        setSimStatus("Shot skipped · calibrate the simulator first");
        if (persist({ ...r, cursor: event.seq })) {
          socket.send(JSON.stringify({ type: "ack", seq: event.seq }));
          pending.shift();
        }
        return;
      }
      if (takeShot(event)) {
        socket.send(JSON.stringify({ type: "ack", seq: event.seq }));
        pending.shift();
        setSimStatus("Connected · shot received");
      }
    }, 200);
    return () => {
      stopped = true;
      clearTimeout(retry);
      clearInterval(heartbeat);
      socket?.close();
    };
  }, [simEnabled, simUrl, persist, takeShot]);
  const askCaddie = () => {
    if (busy || !detailReady || caddieBusy) return;
    worker.current?.terminate();
    const requestBall = round.ball.join(","),
      requestHole = round.hole;
    const w = new Worker(new URL("./caddie.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.current = w;
    setCaddieBusy(true);
    setCaddieNote("Reading the slope…");
    w.onmessage = (e) => {
      setCaddieBusy(false);
      w.terminate();
      if (
        roundRef.current.hole !== requestHole ||
        roundRef.current.ball.join(",") !== requestBall
      )
        return;
      setPower(e.data.power);
      setAim(e.data.aim);
      setCaddieNote(
        e.data.error < 0
          ? "A promising line. Trust the pace."
          : `Suggested shot · about ${Math.max(0, e.data.error).toFixed(1)} m from the pin`,
      );
    };
    w.onerror = () => {
      setCaddieBusy(false);
      setCaddieNote("Could not read this lie.");
      w.terminate();
    };
    w.postMessage({ world, ball: round.ball, club: CLUBS[club] });
  };
  const movePracticeBall = (kind: "putt" | "approach" | "full") => {
    if (busy) return;
    let ball: Vec3 = [...hole.tee];
    if (kind === "putt")
      ball = [
        hole.pin[0],
        heightAt(world, hole.pin[0], hole.pin[2] + 4),
        hole.pin[2] + 4,
      ];
    if (kind === "full" || kind === "approach") {
      const metres = kind === "full" ? 280 : 65;
      const b = bearingTo(hole.pin, hole.tee),
        x = hole.pin[0] + Math.sin(b) * metres,
        z = hole.pin[2] - Math.cos(b) * metres;
      ball = [x, heightAt(world, x, z), z];
    }
    const scores = [...round.scores];
    scores[round.hole] = null;
    if (
      persist({
        ...round,
        ball,
        strokes: 0,
        scores,
        complete: false,
        last: undefined,
      })
    ) {
      setAnimation(undefined);
      swingAPI.current.reset();
      setStrikeFeedback('');
      setCameraMode('player');
      setClub(kind === "putt" ? 13 : kind === "full" ? 0 : 9);
      setPower(kind==='putt'?Math.round(Math.sqrt(2*9.81*.56/stimp*4)/CLUBS[13].speed*100):90);
      setAim(kind === "full" ? openingAim(course,{...round,ball,strokes:0}) : 0);
      setShowSettings(false);
    }
  };
  useEffect(() => {
    if (!(showScore || showHelp || showSettings)) return;
    const previous = document.activeElement as HTMLElement | null;
    const dialog = document.querySelector<HTMLElement>('[role="dialog"]');
    const focusable = () =>
      Array.from(
        dialog?.querySelectorAll<HTMLElement>(
          "button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled)",
        ) ?? [],
      );
    focusable()[0]?.focus();
    const trap = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const items = focusable(),
        first = items[0],
        last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    window.addEventListener("keydown", trap);
    return () => {
      window.removeEventListener("keydown", trap);
      previous?.focus();
    };
  }, [showScore, showHelp, showSettings]);
  const total = round.scores.reduce<number>((sum, s) => sum + (s ?? 0), 0);
  const played = round.scores.filter((s) => s !== null).length;
  const playedPar = round.scores.reduce<number>(
    (sum, s, i) => sum + (s !== null ? course.practice[i].par : 0),
    0,
  );
  const holeComplete = round.scores[round.hole] !== null;
  const bearing = bearingTo(round.ball, hole.pin) + (aim * Math.PI) / 180;
  const canShoot =
    detailReady && !busy && !round.complete && !holeComplete && mode === "golf" && !showMap && !showHelp && !showScore && !showSettings;
  const preview=useMemo(()=>{try{return simulate(world,round.ball,{speed:CLUBS[club].speed*power/100,bearing,launch:CLUBS[club].loft,spin:CLUBS[club].spin}).end;}catch{return round.ball;}},[world,round.ball,club,power,bearing]);
  const relativeScore=total-playedPar;
  return (
    <main className="grenland-app">
      <div className="course-viewport">
        <ErrorBoundary fallback={<div className="course-loading"><h2>The 3D course could not start</h2><p>Use a browser with WebGL 2 enabled, then reload.</p><button onClick={()=>location.reload()}>Reload course</button></div>}>
        <Suspense
          fallback={<div className="course-loading">Preparing the course…</div>}
        >
          <Scene
            course={course}
            world={world}
            ball={round.ball}
            bearing={bearing}
            result={animation}
            motion={motion}
            mode={mode}
            input={controls}
            onSettled={settled}
            onReady={sceneLoaded}
            onTravelDistance={setTravelDistance}
            quality={quality}
            swing={swing.controller}
            club={club}
            cameraMode={cameraMode}
            preview={preview}
            journey={journey}
            showGrid={showGrid}
          />
        </Suspense>
        </ErrorBoundary>
        {!sceneReady&&<div className="course-preparing" role="status">Preparing golfer and course…</div>}
      </div>
      {showMap&&<CourseMap current={exploreHole??round.hole+1} onClose={closeMap}/>}
      <header className="course-topbar">
        <a className="course-brand" href="/play/grenland">
          STRIKE<span>LAB</span>
          <small>GOLF, WITHOUT LIMITS</small>
        </a>
        <div className="course-place">
          <span className="course-eyebrow">NORWAY · 59.27° N</span>
          <strong>Grenland</strong>
          <span>
            {course.mode === "preview"
              ? "18-hole preview · community routing"
              : "Terrain practice · routing under review"}
          </span>
        </div>
        <nav aria-label="Course tools"><button disabled={busy} onClick={()=>setShowMap(true)}>Course map</button>
          <button
            onClick={() => setShowHelp(true)}
            aria-label="Controls and help"
          >
            ?
          </button>
          <button onClick={() => setShowSettings(true)}>Settings</button>
          <button
            onClick={() => {
              setView("practice");
              setShowScore(true);
            }}
          >
            Scorecard
          </button>
        </nav>
      </header>
      <section className="course-hole-card">
        <span className="course-eyebrow">
          {round.complete
            ? exploreHole?"HOLE COMPLETE":"ROUND COMPLETE"
            : `${course.mode === "preview" ? "HOLE" : "PRACTICE"} ${String(exploreHole??round.hole + 1).padStart(2, "0")} / ${exploreHole?18:course.practice.length}`}
        </span>
        <h1>{hole.name}</h1><HoleLinks current={exploreHole??round.hole+1}/>
        <div className="course-hole-stats">
          <span>
            PAR <b>{hole.par}</b>
          </span>
          <span>
            STROKE <b>{round.strokes + (!holeComplete ? 1 : 0)}</b>
          </span>
          <span>
            {played
              ? `${total - playedPar > 0 ? "+" : ""}${total - playedPar}`
              : "E"}{" "}
            <small>ROUND</small>
          </span>
        </div>
        <MiniMap course={course} round={round} />
        <div className="course-map-caption">
          <span>● BALL</span>
          <span>⚑ PRACTICE PIN</span>
        </div>
      </section>
      <div className="course-round-badge"><span>YOUR ROUND</span><strong>{relativeScore===0?"E":`${relativeScore>0?"+":""}${relativeScore}`}</strong><small>{played} THRU · {total+(!holeComplete?round.strokes:0)} STROKES</small></div>
      <div className="course-distance">
        <span className="course-eyebrow">TO THE PIN</span>
        <strong>
          {distance(round.ball, hole.pin).toFixed(1)}
          <small> m</small>
        </strong>
        <span>
          {surfaceAt(world, round.ball[0], round.ball[2])} ·{" "}
          {wind ? `${wind} m/s crosswind` : "Calm wind"}
        </span>
      </div>
      <div className="course-travel" role="group" aria-label="Movement mode">
        {(["golf", "walk", "cart"] as const).map((m) => (
          <button
            key={m}
            className={mode === m ? "active" : ""}
            disabled={busy}
            onClick={() => setMode(m)}
          >
            {m === "golf" ? "Play golf" : m === "walk" ? "Walk" : "Drive cart"}
          </button>
        ))}
      </div>
      <div className="course-view-tools"><button onClick={()=>setCameraMode(v=>v==="player"?"scout":"player")} className={cameraMode==="scout"?"active":""}>Scout view · V</button><button onClick={()=>setShowGrid(v=>!v)} aria-pressed={showGrid}>Green grid · G</button></div>
      {error && (
        <div className="course-alert" role="alert">
          {error}
          <button onClick={() => setError("")} aria-label="Dismiss error">
            ×
          </button>
        </div>
      )}
      {(!detailReady || busy) && (
        <div className="course-status" role="status">
          {!detailReady ? "Loading green detail…" : "Ball in motion"}
        </div>
      )}
      {holeComplete && !busy && (
        <section className="course-result" aria-live="polite">
          <span className="course-eyebrow">
            {round.complete ? exploreHole?"HOLE COMPLETE":"ALL GREENS COMPLETED" : "IN THE CUP"}
          </span>
          <h2>
            {round.complete
              ? exploreHole?scoreName(round.strokes,hole.par):"Round complete"
              : scoreName(round.strokes, hole.par)}
          </h2>
          <p>
            {round.complete
              ? `${total} strokes across ${played} holes.`
              : "Take a moment. Then find your next green."}
          </p>
          <button
            className="course-primary"
            onClick={round.complete ? () => setShowScore(true) : nextHole}
          >
            {round.complete ? "View final scorecard" : "Next tee →"}
          </button>
          {!round.complete&&<button className="course-walk-next" onClick={()=>{setJourney([...round.ball]);nextHole();setMode("walk");}}>Walk to next tee</button>}
        </section>
      )}
      {mode === "golf" ? (
        <section className="course-shot-panel" aria-label="Shot controls">
          <label className="course-club">
            <span className="course-eyebrow">YOUR CLUB</span>
            <select
              aria-label="Club"
              value={club}
              disabled={busy}
              onChange={(e) => setClub(Number(e.target.value))}
            >
              {CLUBS.map((c, i) => (
                <option key={c.name} value={i}>
                  {c.name}
                </option>
              ))}
            </select>
            <small>
              {CLUBS[club].loft}° launch ·{" "}
              {CLUBS[club].name === "Putter"
                ? "Feel the green"
                : "Play your line"}
            </small>
            <button
              className="course-caddie"
              disabled={!canShoot || caddieBusy}
              onClick={askCaddie}
            >
              {caddieBusy ? "Reading…" : "Caddie suggestion"}
            </button>
          </label>
          <label className="course-slider">
            <span>
              AIM{" "}
              <b>
                {aim > 0 ? "+" : ""}
                {aim.toFixed(1)}°
              </b>
            </span>
            <input
              type="range"
              aria-label="Aim angle"
              min="-180"
              max="180"
              step="0.1"
              value={aim}
              disabled={busy}
              onChange={(e) => setAim(Number(e.target.value))}
            />
            <small>← Left / right →</small>
          </label>
          <label className="course-slider">
            <span>
              TARGET POWER <b>{Math.round(power)}%</b>
            </span>
            <input
              type="range"
              aria-label="Shot power"
              min="1"
              max="100"
              step="1"
              value={power}
              disabled={busy}
              onChange={(e) => setPower(Number(e.target.value))}
            />
            <small>Estimated {distance(round.ball,preview).toFixed(0)} m total</small>
          </label>
          <SwingPanel swing={swing} mode={swingMode} disabled={!canShoot} busy={busy}/>
        </section>
      ) : (
        <section className="course-movement-panel">
          <span>
            {mode === "walk" ? "Explore on foot" : "Explore by cart"} · WASD /
            left stick · {Math.round(travelDistance)} m to next shot
          </span>
          <DriveStick disabled={showMap||showHelp||showScore||showSettings} onMove={axes=>{touch.current=axes;}}/>
          {mode==='cart'&&<button className="drive-boost" aria-label="Toggle boost" aria-pressed={boost} onClick={()=>setBoost(v=>!v)}>⚡ Boost {boost?'on':'off'} · Shift / R2</button>}
          <button onClick={() => setMode("golf")}>Return to ball</button>
        </section>
      )}
      {strikeFeedback&&!busy&&!caddieNote&&!holeComplete&&<div className="course-strike-feedback">{strikeFeedback}</div>}
      {caddieNote && !busy && (
        <div className="course-caddie-note" role="status">
          {caddieNote}
        </div>
      )}
      {round.last && !busy && !caddieNote && !holeComplete && (
        <div className="course-caddie-note" role="status">
          {round.last.penalty
            ? `${round.last.reason} · +1 penalty stroke`
            : `Last shot · ${round.last.carry > 0 ? `${round.last.carry.toFixed(1)} m carry · ` : ""}${round.last.distance.toFixed(1)} m total`}
        </div>
      )}
      <footer className="course-footer">
        <span className={simEnabled ? "sim-active" : ""}>
          ● {simEnabled ? simStatus : "Standalone play · no simulator needed"}
        </span>
        <span>
          {pad ? "Controller connected" : "Touch · mouse · keyboard · gamepad"}
        </span>
        <span>
          Kartverket terrain ·{" "}
          <a
            href="https://www.openstreetmap.org/copyright"
            target="_blank"
            rel="noreferrer"
          >
            Routing © OpenStreetMap
          </a>
        </span>
      </footer>
      {showScore && (
        <div className="course-modal-backdrop">
          <section
            className="course-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Scorecard"
          >
            <button
              className="course-close"
              onClick={() => setShowScore(false)}
              aria-label="Close scorecard"
            >
              ×
            </button>
            <span className="course-eyebrow">GRENLAND</span>
            <h2>
              {view === "practice" ? "Your round" : "Official course reference"}
            </h2>
            <div className="course-tabs">
              <button
                className={view === "practice" ? "active" : ""}
                onClick={() => setView("practice")}
              >
                Your scorecard
              </button>
              <button
                className={view === "official" ? "active" : ""}
                onClick={() => setView("official")}
              >
                Club guide reference
              </button>
            </div>
            {view === "practice" ? (
              <>
                <p>
                  {played} / {course.practice.length} completed · {total}{" "}
                  strokes · saved on this device
                </p>
                <div className="course-score-grid">
                  {round.scores.map((s, i) => (
                    <div key={i} className={i === round.hole ? "current" : ""}>
                      <span>{String(i + 1).padStart(2, "0")}</span>
                      <strong>{s ?? "—"}</strong>
                      <small>PAR {course.practice[i].par}</small>
                    </div>
                  ))}
                </div>
                <button
                  disabled={busy}
                  onClick={() => {
                    const next = newRound(course);
                    next.cursor = round.cursor;
                    next.device = round.device;
                    if (persist(next)) {
                      setAnimation(undefined);
                      setShowScore(false);
                      setAim(openingAim(course,next));
                      setPower(90);
                      setClub(course.mode === "preview" ? 0 : 9);
                    }
                  }}
                >
                  Start a new round
                </button>
              </>
            ) : (
              <>
                <p>
                  The club guide below uses older tee labels and sums to par 71.
                  The club’s rating card valid through 2026 states par 72. The
                  18-hole preview uses community-mapped routes and pars; its
                  tees and synthetic pins still require club verification.
                </p>
                <div className="course-table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Hole</th>
                        <th>Name</th>
                        <th>Par</th>
                        <th>Index</th>
                        {["59", "57", "53", "48"].map((t) => (
                          <th key={t}>T{t}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {course.holes.map((h) => (
                        <tr key={h.id}>
                          <td>{h.number}</td>
                          <td>{h.name}</td>
                          <td>{h.par}</td>
                          <td>{h.index}</td>
                          {["59", "57", "53", "48"].map((t) => (
                            <td key={t}>{h.tees_m[t]}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <a href={course.reference_url} target="_blank" rel="noreferrer">
                  Open the club’s course guide ↗
                </a>
              </>
            )}
          </section>
        </div>
      )}
      {showHelp && (
        <div className="course-modal-backdrop">
          <section
            className="course-modal course-modal-small"
            role="dialog"
            aria-modal="true"
            aria-label="Controls"
          >
            <button
              className="course-close"
              onClick={() => setShowHelp(false)}
              aria-label="Close help"
            >
              ×
            </button>
            <span className="course-eyebrow">MAKE YOURSELF AT HOME</span>
            <h2>Your game. Your controls.</h2>
            <dl>
              <dt>Mouse & keyboard</dt>
              <dd>
                Drag the view to look, scroll to zoom. Drag down then up on the swing pad to hit. Space uses three-click timing. Arrow keys adjust aim and target power. V scouts the landing area; G toggles the green grid.
                C changes club, N advances after holing out. WASD
                moves in walk/cart mode.
              </dd>
              <dt>Touch</dt>
              <dd>
                Drag the view, pinch to zoom. Use the aim and power sliders,
                then pull down and smoothly return on the swing pad. Three-click timing is available in Settings. Direction buttons move you when exploring.
              </dd>
              <dt>PS5 / standard gamepad</dt>
              <dd>
                Connect by USB or Bluetooth, then press a button. Left stick
                aims and changes power; right stick pulls back and swings through. L1/R1 change club. ✕ uses timing, □ scouts, ○ advances, △
                changes movement mode, Options opens the scorecard.
              </dd>
              <dt>Real simulator</dt>
              <dd>
                Optional in Settings. Calibrate the local backend first. Its
                measured putts feed this game; the course determines the
                outcome. Full-swing and measured spin hardware are not yet
                verified.
              </dd>
            </dl>
            <p>
              Practice pins and tees are synthetic. Terrain and green contours
              come from the existing Kartverket-derived dataset; flight and roll
              need calibration before use as a measurement instrument.
            </p>
          </section>
        </div>
      )}
      {showSettings && (
        <div className="course-modal-backdrop">
          <section
            className="course-modal course-modal-small"
            role="dialog"
            aria-modal="true"
            aria-label="Course settings"
          >
            <button
              className="course-close"
              onClick={() => setShowSettings(false)}
              aria-label="Close settings"
            >
              ×
            </button>
            <span className="course-eyebrow">SET UP YOUR SESSION</span>
            <h2>Make it your round.</h2>
            <label>
              Course layout
              <select
                aria-label="Course layout"
                value={course.mode ?? "practice"}
                disabled={busy}
                onChange={(e) =>
                  onCourseMode(e.target.value as "preview" | "practice")
                }
              >
                <option value="preview">18-hole Grenland preview</option>
                <option value="practice">25 practice greens</option>
              </select>
            </label>
            <p>
              Community routes follow 18 numbered holes. Practice pins are
              synthetic; survey and club verification remain open. Each layout
              saves its own round.
            </p>
            <label>Swing controls<select aria-label="Swing controls" value={swingMode} disabled={busy} onChange={e=>setSwingMode(e.target.value as SwingMode)}><option value="analog">Analog · mouse / touch / stick</option><option value="three-click">Three-click timing</option></select></label>
            <label>
              Green speed · Stimp {stimp}
              <input
                type="range"
                min="6"
                max="14"
                step=".5"
                value={stimp}
                disabled={busy}
                onChange={(e) => {
                  const n = Number(e.target.value);
                  setStimp(n);
                  try {
                    localStorage.setItem(PREFS, JSON.stringify({ stimp: n }));
                  } catch {
                    /* Round saving handles storage errors. */
                  }
                }}
              />
            </label>
            <label>
              Crosswind · {wind} m/s
              <input
                type="range"
                min="-8"
                max="8"
                step=".5"
                value={wind}
                disabled={busy}
                onChange={(e) => setWind(Number(e.target.value))}
              />
            </label>
            <label>
              Graphics
              <select
                value={quality}
                onChange={(e) =>
                  setQuality(e.target.value as "balanced" | "high")
                }
              >
                <option value="balanced">Balanced · mobile friendly</option>
                <option value="high">High resolution</option>
              </select>
            </label>
            <button
              aria-pressed={sound}
              onClick={() => {
                setSound((v) => !v);
                try {
                  localStorage.setItem(
                    "strikelab.golf.sound",
                    sound ? "off" : "on",
                  );
                } catch {
                  /* Optional preference. */
                }
              }}
            >
              Sound {sound ? "on" : "off"}
            </button>
            <fieldset>
              <legend>Restart this green from a practice position</legend>
              <div className="course-tabs">
                <button
                  disabled={busy}
                  onClick={() => movePracticeBall("putt")}
                >
                  4 m putt
                </button>
                <button
                  disabled={busy}
                  onClick={() => movePracticeBall("approach")}
                >
                  65 m approach
                </button>
                <button
                  disabled={busy}
                  onClick={() => movePracticeBall("full")}
                >
                  280 m tee
                </button>
              </div>
            </fieldset>
            <fieldset>
              <legend>Optional physical simulator</legend>
              <label>
                Launch stream address
                <input
                  aria-label="Simulator WebSocket address"
                  value={simUrl}
                  disabled={simEnabled}
                  onChange={(e) => {
                    setSimUrl(e.target.value);
                    try {
                      localStorage.setItem("strikelab.sim.url", e.target.value);
                    } catch {
                      /* Optional preference. */
                    }
                  }}
                  placeholder="ws://computer-address:8000/ws/shots"
                />
              </label>
              <button onClick={() => setSimEnabled((v) => !v)}>
                {simEnabled ? "Disconnect simulator" : "Connect simulator"}
              </button>
              <button
                disabled={busy}
                onClick={() => {
                  setSimEnabled(false);
                  persist({
                    ...roundRef.current,
                    cursor: undefined,
                    device: undefined,
                  });
                  setSimStatus(
                    "Pairing cleared. Connect to the chosen simulator.",
                  );
                }}
              >
                Forget simulator pairing
              </button>
              <p role="status">{simStatus}</p>
              <small>
                On a phone, use the simulator computer’s LAN address. HTTPS
                pages require a secure wss:// endpoint. Remote relay pairing is
                not configured.
              </small>
            </fieldset>
          </section>
        </div>
      )}
    </main>
  );
}

export default function CourseApp() {
  const requested=Number(new URLSearchParams(location.search).get('hole'));
  const exploreHole=Number.isInteger(requested)&&requested>=1&&requested<=18?requested:undefined;
  const [data, setData] = useState<{
      course: Course;
      terrain: Tile;
      preview: PracticeHole[];
      previewRevision: string;
    }>(),
    [error, setError] = useState("");
  const [courseMode, setCourseMode] = useState<"preview" | "practice">(() => {
    try {
      return localStorage.getItem("strikelab.course.mode") === "practice"
        ? "practice"
        : "preview";
    } catch {
      return "preview";
    }
  });
  const changeCourse = (mode: "preview" | "practice") => {
    setCourseMode(mode);
    try {
      localStorage.setItem("strikelab.course.mode", mode);
    } catch {
      /* Optional preference. */
    }
  };
  const activeCourse = useMemo(
    () =>
      data
        ? {
            ...data.course,
            mode: exploreHole?"preview" as const:courseMode,
            practice:
              exploreHole?[data.preview[exploreHole-1]]:courseMode === "preview" ? data.preview : data.course.practice,
            revision: exploreHole?`${data.previewRevision}-explore-${exploreHole}`:
              courseMode === "preview"
                ? data.previewRevision
                : data.course.revision,
          }
        : undefined,
    [data, courseMode, exploreHole],
  );
  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      const response = await fetch(COURSE_ROOT + "manifest.json", {
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("Course manifest could not be loaded");
      const course: Course = await response.json();
      if (course.version !== 1 || !course.practice?.length)
        throw new Error("Unsupported course data");
      const [terrain, routesResponse] = await Promise.all([
        loadTile(course.terrain, controller.signal),
        fetch(COURSE_ROOT + "routes.json", { signal: controller.signal }),
      ]);
      if (!routesResponse.ok)
        throw new Error("Course routes could not be loaded");
      const routes = await routesResponse.json();
      if (routes.version !== 1 || routes.preview?.length !== 18)
        throw new Error("Unsupported course routing");
      setData({
        course,
        terrain,
        preview: routes.preview,
        previewRevision: routes.revision,
      });
    })().catch((e) => {
      if (e.name !== "AbortError") setError(e.message);
    });
    return () => controller.abort();
  }, []);
  return data && activeCourse ? (
    <Game
      key={`${courseMode}-${exploreHole??"round"}`}
      exploreHole={exploreHole}
      course={activeCourse}
      terrain={data.terrain}
      onCourseMode={changeCourse}
    />
  ) : (
    <main className="grenland-app course-loading">
      <span className="course-eyebrow">STRIKELAB / GRENLAND</span>
      <h1>
        {error ? "The course could not load" : "Your next round starts here."}
      </h1>
      <p>{error || "Preparing the landscape…"}</p>
      {error && <button onClick={() => location.reload()}>Try again</button>}
    </main>
  );
}
