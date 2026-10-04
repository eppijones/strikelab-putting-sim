// SI units. Shared terrain sampling for rendered triangles and golf collision.
export type Vec3 = [number, number, number];
export interface TileSpec {
  url: string;
  surfaces: string;
  size: number;
  spacing: number;
  origin: [number, number];
  texture?: string;
}
export interface Tile extends TileSpec {
  heights: Float32Array;
  lies: Uint8Array;
}
export interface PracticeHole {
  id: string;
  name: string;
  par: number;
  pin: Vec3;
  tee: Vec3;
  tile: TileSpec;
  route?: [number, number][];
}
export interface OfficialHole {
  id: string;
  number: number;
  name: string;
  par: number;
  index: number;
  tees_m: Record<string, number>;
  routing_status: string;
}
export interface Region {
  id: string;
  kind: string;
  status: string;
  points: [number, number][];
}
export interface Course {
  version: number;
  revision: string;
  name: string;
  terrain: TileSpec;
  practice: PracticeHole[];
  holes: OfficialHole[];
  regions: Region[];
  reference_url: string;
  mode?: "preview" | "practice";
}
export type Surface =
  "rough" | "fairway" | "green" | "bunker" | "water" | "out";
export interface World {
  terrain: Tile;
  detail?: Tile;
  pin: Vec3;
  hazards?: Region[];
  stimp: number;
  wind: [number, number];
}
export interface Shot {
  speed: number;
  bearing: number;
  launch: number;
  spin: number;
}
export interface ShotResult {
  end: Vec3;
  path: Vec3[];
  made: boolean;
  penalty: number;
  reason: string;
  carry: number;
  distance: number;
  duration: number;
}
export interface Club {
  name: string;
  speed: number;
  loft: number;
  spin: number;
}
export const CLUBS: Club[] = [
  { name: "Driver", speed: 70, loft: 12, spin: 2500 },
  { name: "3 wood", speed: 61, loft: 15, spin: 3300 },
  { name: "5 wood", speed: 56, loft: 19, spin: 4000 },
  { name: "4 iron", speed: 51, loft: 21, spin: 4500 },
  { name: "5 iron", speed: 48, loft: 24, spin: 5000 },
  { name: "6 iron", speed: 44, loft: 28, spin: 5600 },
  { name: "7 iron", speed: 40, loft: 32, spin: 6200 },
  { name: "8 iron", speed: 36, loft: 36, spin: 7000 },
  { name: "9 iron", speed: 33, loft: 41, spin: 7600 },
  { name: "PW", speed: 29, loft: 46, spin: 8200 },
  { name: "GW", speed: 25, loft: 50, spin: 8500 },
  { name: "SW", speed: 22, loft: 56, spin: 9000 },
  { name: "LW", speed: 18, loft: 60, spin: 9500 },
  { name: "Putter", speed: 5, loft: 0, spin: 0 },
];
export const distance = (a: Vec3, b: Vec3) =>
  Math.hypot(a[0] - b[0], a[2] - b[2]);
export const bearingTo = (a: Vec3, b: Vec3) =>
  Math.atan2(b[0] - a[0], -(b[2] - a[2]));
export function contains(tile: TileSpec, x: number, z: number) {
  return (
    x >= tile.origin[0] &&
    z >= tile.origin[1] &&
    x <= tile.origin[0] + (tile.size - 1) * tile.spacing &&
    z <= tile.origin[1] + (tile.size - 1) * tile.spacing
  );
}
export function sample(tile: Tile, x: number, z: number): number {
  const u = Math.max(
    0,
    Math.min(tile.size - 1.000001, (x - tile.origin[0]) / tile.spacing),
  );
  const v = Math.max(
    0,
    Math.min(tile.size - 1.000001, (z - tile.origin[1]) / tile.spacing),
  );
  const ix = Math.floor(u),
    iz = Math.floor(v),
    fx = u - ix,
    fz = v - iz,
    a = iz * tile.size + ix;
  const h = tile.heights;
  // The mesh splits each quad on its NE-SW diagonal.
  return fx + fz <= 1
    ? h[a] + fx * (h[a + 1] - h[a]) + fz * (h[a + tile.size] - h[a])
    : h[a + tile.size + 1] +
        (1 - fx) * (h[a + tile.size] - h[a + tile.size + 1]) +
        (1 - fz) * (h[a + 1] - h[a + tile.size + 1]);
}
export function heightAt(world: World, x: number, z: number) {
  return sample(
    world.detail && contains(world.detail, x, z) ? world.detail : world.terrain,
    x,
    z,
  );
}
export function inPolygon(x: number, z: number, points: [number, number][]) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const a = points[i],
      b = points[j];
    if (
      a[1] > z !== b[1] > z &&
      x < ((b[0] - a[0]) * (z - a[1])) / (b[1] - a[1]) + a[0]
    )
      inside = !inside;
  }
  return inside;
}
export function surfaceAt(world: World, x: number, z: number): Surface {
  if (!contains(world.terrain, x, z)) return "out";
  if (
    world.hazards?.some((r) => r.kind === "water" && inPolygon(x, z, r.points))
  )
    return "water";
  const tile =
    world.detail && contains(world.detail, x, z) ? world.detail : world.terrain;
  const ix = Math.round((x - tile.origin[0]) / tile.spacing),
    iz = Math.round((z - tile.origin[1]) / tile.spacing);
  return (
    (["rough", "fairway", "green", "bunker"] as const)[
      tile.lies[iz * tile.size + ix]
    ] ?? "rough"
  );
}
export function simulate(world: World, start: Vec3, shot: Shot): ShotResult {
  if (
    ![
      ...start,
      shot.speed,
      shot.bearing,
      shot.launch,
      shot.spin,
      world.stimp,
      ...world.wind,
    ].every(Number.isFinite) ||
    shot.speed <= 0 ||
    shot.speed > 100 ||
    shot.launch < 0 ||
    shot.launch > 80 ||
    world.stimp < 4 ||
    world.stimp > 18
  )
    throw new Error("Invalid launch or course conditions");
  const dt = 1 / 240,
    radius = 0.02135,
    p: Vec3 = [...start];
  p[1] = heightAt(world, p[0], p[2]) + radius;
  const angle = (shot.launch * Math.PI) / 180;
  const v: Vec3 = [
    Math.sin(shot.bearing) * shot.speed * Math.cos(angle),
    shot.speed * Math.sin(angle),
    -Math.cos(shot.bearing) * shot.speed * Math.cos(angle),
  ];
  let airborne = shot.launch > 0,
    carry = 0,
    made = false,
    elapsed = 0,
    penalty = 0,
    reason = "At rest";
  let stopped = false;
  const path: Vec3[] = [[...p]];
  for (let tick = 0; tick < 240 * 60; tick++) {
    const old: Vec3 = [...p];
    const lie = surfaceAt(world, p[0], p[2]);
    if ((lie === "water" && !airborne) || lie === "out") {
      penalty = 1;
      reason =
        lie === "water"
          ? "Water · stroke and distance"
          : "Out of bounds · stroke and distance";
      break;
    }
    if (airborne) {
      const air: Vec3 = [v[0] - world.wind[0], v[1], v[2] - world.wind[1]];
      const speed = Math.hypot(...air),
        drag = 0.0065 * speed;
      // Approximate drag and backspin lift. Deliberately not a calibrated launch-monitor model.
      const lift =
        0.0012 * speed * speed * Math.min(1.5, Math.abs(shot.spin) / 3000);
      v[0] -= air[0] * drag * dt;
      v[2] -= air[2] * drag * dt;
      v[1] += (lift - 9.81 - air[1] * drag) * dt;
      p[0] += v[0] * dt;
      p[1] += v[1] * dt;
      p[2] += v[2] * dt;
      const ground = heightAt(world, p[0], p[2]) + radius;
      if (p[1] <= ground && v[1] < 0) {
        if (!carry) carry = distance(start, p);
        p[1] = ground;
        const impactLie = surfaceAt(world, p[0], p[2]);
        if (impactLie === "water") {
          penalty = 1;
          reason = "Water · stroke and distance";
          break;
        }
        const bounce =
          impactLie === "bunker" ? 0.08 : impactLie === "rough" ? 0.16 : 0.3;
        v[1] = -v[1] * bounce;
        v[0] *= impactLie === "bunker" ? 0.35 : 0.73;
        v[2] *= impactLie === "bunker" ? 0.35 : 0.73;
        if (v[1] < 0.55) {
          airborne = false;
          v[1] = 0;
        }
      }
    } else {
      const eps = 0.15,
        gx =
          (heightAt(world, p[0] + eps, p[2]) -
            heightAt(world, p[0] - eps, p[2])) /
          (2 * eps);
      const gz =
        (heightAt(world, p[0], p[2] + eps) -
          heightAt(world, p[0], p[2] - eps)) /
        (2 * eps);
      const friction =
        lie === "green"
          ? (9.81 * 0.56) / world.stimp
          : lie === "fairway"
            ? 1.25
            : lie === "bunker"
              ? 5
              : 2.8;
      const slope = (Math.hypot(gx, gz) * 9.81 * 5) / 7;
      const speed = Math.hypot(v[0], v[2]);
      if (speed < 0.035 && slope < friction) {
        stopped = true;
        break;
      }
      v[0] -= ((gx * 9.81 * 5) / 7) * dt;
      v[2] -= ((gz * 9.81 * 5) / 7) * dt;
      const s = Math.hypot(v[0], v[2]),
        factor = Math.max(0, 1 - (friction * dt) / Math.max(s, 1e-8));
      v[0] *= factor;
      v[2] *= factor;
      p[0] += v[0] * dt;
      p[2] += v[2] * dt;
      p[1] = heightAt(world, p[0], p[2]) + radius;
    }
    // Swept cup capture prevents skipping a small cup between integration steps.
    const dx = p[0] - old[0],
      dz = p[2] - old[2];
    const t = Math.max(
      0,
      Math.min(
        1,
        ((world.pin[0] - old[0]) * dx + (world.pin[2] - old[2]) * dz) /
          (dx * dx + dz * dz || 1),
      ),
    );
    if (
      !airborne &&
      Math.hypot(
        old[0] + dx * t - world.pin[0],
        old[2] + dz * t - world.pin[2],
      ) <
        0.054 - radius &&
      Math.hypot(v[0], v[2]) < 1.5
    ) {
      made = true;
      reason = "Holed";
      p[0] = world.pin[0];
      p[1] = world.pin[1];
      p[2] = world.pin[2];
      stopped = true;
      break;
    }
    elapsed += dt;
    if (tick % 8 === 0) path.push([...p]);
  }
  if (!stopped && !penalty)
    reason = "Simulation limit · ball placed at last position";
  path.push([...p]);
  const end: Vec3 = penalty ? [...start] : [...p];
  return {
    end,
    path,
    made,
    penalty,
    reason,
    carry,
    distance: distance(start, p),
    duration: elapsed,
  };
}

export interface Round {
  version: 1;
  revision: string;
  id: string;
  hole: number;
  ball: Vec3;
  strokes: number;
  scores: (number | null)[];
  complete: boolean;
  cursor?: number;
  device?: string;
  accepted: string[];
  last?: ShotResult;
}
export function scoreName(strokes: number, par: number): string {
  if (strokes === 1) return "Hole in one!";
  return (
    (
      {
        "-3": "Albatross",
        "-2": "Eagle",
        "-1": "Birdie",
        "0": "Par",
        "1": "Bogey",
        "2": "Double bogey",
      } as Record<string, string>
    )[String(strokes - par)] ?? `${strokes} strokes`
  );
}
export function sessionId(): string {
  // getRandomValues also works on an HTTP LAN origin; randomUUID requires HTTPS.
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64;
  bytes[8] = (bytes[8] & 63) | 128;
  const h = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}
export function newRound(course: Course): Round {
  return {
    version: 1,
    revision: course.revision,
    id: sessionId(),
    hole: 0,
    ball: [...course.practice[0].tee],
    strokes: 0,
    scores: course.practice.map(() => null),
    complete: false,
    accepted: [],
  };
}
export function applyShot(
  round: Round,
  result: ShotResult,
  eventId?: string,
): Round {
  if (
    round.complete ||
    round.scores[round.hole] !== null ||
    (eventId && round.accepted.includes(eventId))
  )
    return round;
  const strokes = round.strokes + 1 + result.penalty,
    scores = [...round.scores];
  if (result.made) scores[round.hole] = strokes;
  return {
    ...round,
    ball: result.end,
    strokes,
    scores,
    complete: scores.every((s) => s !== null),
    last: result,
    accepted: eventId ? [...round.accepted, eventId] : round.accepted,
  };
}
export function advance(round: Round, course: Course): Round {
  if (round.scores[round.hole] === null || round.complete) return round;
  const hole = round.hole + 1;
  if (hole >= course.practice.length) return { ...round, complete: true };
  return {
    ...round,
    hole,
    ball: [...course.practice[hole].tee],
    strokes: 0,
    last: undefined,
  };
}
export function restoreRound(raw: string | null, course: Course): Round | null {
  try {
    const r: Round = JSON.parse(raw ?? "null");
    if (
      !r ||
      r.version !== 1 ||
      r.revision !== course.revision ||
      !Number.isInteger(r.hole) ||
      r.hole < 0 ||
      r.hole >= course.practice.length ||
      !Array.isArray(r.ball) ||
      r.ball.length !== 3 ||
      !r.ball.every(Number.isFinite) ||
      !Number.isInteger(r.strokes) ||
      r.strokes < 0 ||
      r.strokes > 1000 ||
      !Array.isArray(r.scores) ||
      r.scores.length !== course.practice.length ||
      !r.scores.every((s) => s === null || (Number.isInteger(s) && s > 0)) ||
      !Array.isArray(r.accepted) ||
      !r.accepted.every((s) => typeof s === "string") ||
      typeof r.id !== "string" ||
      r.id.length < 8 ||
      typeof r.complete !== "boolean" ||
      r.complete !== r.scores.every((s) => s !== null) ||
      r.scores.some((s, i) =>
        i < r.hole ? s === null : i > r.hole ? s !== null : false,
      ) ||
      (r.scores[r.hole] !== null && r.scores[r.hole] !== r.strokes) ||
      (r.device !== undefined && typeof r.device !== "string") ||
      (r.cursor !== undefined &&
        (!Number.isSafeInteger(r.cursor) || r.cursor < 0))
    )
      return null;
    if (
      r.last &&
      (!Array.isArray(r.last.end) ||
        !r.last.end.every(Number.isFinite) ||
        !Array.isArray(r.last.path) ||
        !r.last.path.every(
          (p) => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite),
        ) ||
        ![r.last.carry, r.last.distance, r.last.duration, r.last.penalty].every(
          Number.isFinite,
        ) ||
        typeof r.last.made !== "boolean" ||
        typeof r.last.reason !== "string")
    )
      return null;
    return r;
  } catch {
    return null;
  }
}
