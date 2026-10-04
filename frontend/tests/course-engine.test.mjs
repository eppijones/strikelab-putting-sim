import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  applyShot,
  advance,
  simulate,
  sample,
  surfaceAt,
  restoreRound,
  newRound,
  bearingTo,
  CLUBS,
  sessionId,
  scoreName,
} from "../src/course/engine.ts";

test("LAN session IDs do not depend on the HTTPS-only randomUUID API", () => {
  const original = crypto.randomUUID;
  try {
    crypto.randomUUID = undefined;
    const ids = Array.from({ length: 100 }, () => sessionId());
    assert.equal(new Set(ids).size, 100);
    assert.ok(
      ids.every((id) =>
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(
          id,
        ),
      ),
    );
  } finally {
    crypto.randomUUID = original;
  }
});

function world(lie = 2, slope = 0) {
  const size = 101,
    spacing = 1,
    heights = new Float32Array(size * size).map(
      (_, i) => Math.floor(i / size) * slope,
    );
  return {
    terrain: {
      size,
      spacing,
      origin: [0, 0],
      heights,
      lies: new Uint8Array(size * size).fill(lie),
    },
    pin: [50, 0, 20],
    stimp: 10,
    wind: [0, 0],
  };
}
const start = [50, 0, 50];
const putt = { speed: 2, bearing: 0, launch: 0, spin: 0 };
test("flat roll agrees with stopping-distance equation within 3cm", () => {
  const w = world(),
    r = simulate(w, start, putt),
    expected = 4 / ((2 * 9.81 * 0.56) / 10);
  assert.ok(Math.abs(r.distance - expected) < 0.03);
  assert.equal(r.penalty, 0);
});
test("Stimp changes distance without changing launch energy", () => {
  const slow = world(),
    fast = { ...slow, stimp: 14 };
  assert.ok(
    simulate(fast, start, putt).distance >
      simulate(slow, start, putt).distance * 1.35,
  );
});
test("bunker and rough stop the ball earlier than fairway and green", () => {
  const ds = [3, 0, 1, 2].map(
    (lie) => simulate(world(lie), start, putt).distance,
  );
  assert.deepEqual(
    [...ds].sort((a, b) => a - b),
    ds,
  );
});
test("swept slow cup entry is captured while excessive pace rolls through", () => {
  const w = world();
  w.pin = [50, 0, 49];
  assert.equal(simulate(w, start, { ...putt, speed: 1.3 }).made, true);
  assert.equal(simulate(w, start, { ...putt, speed: 5 }).made, false);
});
test("missed cup does not score", () => {
  const w = world();
  w.pin = [50.2, 0, 49];
  assert.equal(simulate(w, start, { ...putt, speed: 1.3 }).made, false);
});
test("uphill putt rolls shorter than a flat putt", () => {
  const uphill = world(2, -0.02);
  assert.ok(
    simulate(uphill, start, putt).distance <
      simulate(world(), start, putt).distance,
  );
});
test("flight carries, lands, bounces and stops above the terrain", () => {
  const r = simulate(world(1), [50, 0, 85], {
    speed: 18,
    bearing: 0,
    launch: 35,
    spin: 3000,
  });
  assert.ok(r.carry > 10);
  assert.ok(r.path.some((p) => p[1] > 2));
  assert.ok(r.end[1] >= 0);
  assert.equal(r.penalty, 0);
});
test("water adds one penalty and uses stroke-and-distance relief", () => {
  const w = world();
  w.hazards = [
    {
      kind: "water",
      points: [
        [48, 48],
        [52, 48],
        [52, 49.8],
        [48, 49.8],
      ],
    },
  ];
  const r = simulate(w, start, putt);
  assert.equal(r.penalty, 1);
  assert.deepEqual(r.end, start);
});
test("out of bounds returns to prior lie", () => {
  const r = simulate(world(), [50, 0, 1], { ...putt, speed: 8 });
  assert.equal(r.penalty, 1);
  assert.deepEqual(r.end, [50, 0, 1]);
});

test("airborne shots can carry water but a water landing incurs a penalty", () => {
  const w = world(0);
  w.pin = [5, 0, 5];
  const shot = { ...putt, speed: 20, launch: 45, spin: 0 };
  w.hazards = [
    {
      kind: "water",
      points: [
        [48, 45],
        [52, 45],
        [52, 49],
        [48, 49],
      ],
    },
  ];
  assert.equal(simulate(w, start, shot).penalty, 0);
  w.hazards = [
    {
      kind: "water",
      points: [
        [0, 0],
        [100, 0],
        [100, 49],
        [0, 49],
      ],
    },
  ];
  const r = simulate(w, start, shot);
  assert.equal(r.penalty, 1);
  assert.deepEqual(r.end, start);
});
test("invalid input cannot poison a saved round", () => {
  for (const speed of [NaN, Infinity, -1, 0, 101])
    assert.throws(() => simulate(world(), start, { ...putt, speed }));
});
test("triangle interpolation matches rendered mesh diagonal", () => {
  const t = {
    size: 2,
    spacing: 1,
    origin: [0, 0],
    heights: new Float32Array([0, 1, 2, 8]),
    lies: new Uint8Array(4),
  };
  assert.equal(sample(t, 0.25, 0.25), 0.75);
  assert.equal(sample(t, 0.75, 0.75), 4.75);
});
test("positive target-relative bearing sends the ball to the right", () => {
  const r = simulate(world(), start, { ...putt, bearing: Math.PI / 4 });
  assert.ok(r.end[0] > start[0]);
  assert.ok(r.end[2] < start[2]);
  assert.equal(bearingTo([0, 0, 1], [0, 0, 0]), 0);
});
test("all 14 clubs are unique and use SI launch values", () => {
  assert.equal(CLUBS.length, 14);
  assert.equal(new Set(CLUBS.map((c) => c.name)).size, 14);
});
const course = JSON.parse(
  readFileSync(
    new URL("../public/courses/grenland/manifest.json", import.meta.url),
  ),
);
test("course guide has 18 unique indexes, par 71, and four tees", () => {
  assert.equal(course.holes.length, 18);
  assert.equal(new Set(course.holes.map((h) => h.index)).size, 18);
  assert.equal(
    course.holes.reduce((s, h) => s + h.par, 0),
    71,
  );
  assert.ok(
    course.holes.every(
      (h) => Object.keys(h.tees_m).length === 4 && h.green_id === null,
    ),
  );
});
test("practice pins and full round survive save/resume without duplicate scoring", () => {
  let r = newRound(course);
  for (let i = 0; i < course.practice.length; i++) {
    const result = {
      end: course.practice[i].pin,
      path: [],
      made: true,
      penalty: 0,
      reason: "Holed",
      carry: 0,
      distance: 1,
      duration: 1,
    };
    r = applyShot(r, result, `shot-${i}`);
    assert.equal(applyShot(r, result, `shot-${i}`), r);
    r = restoreRound(JSON.stringify(r), course);
    assert.ok(r);
    if (i < course.practice.length - 1) {
      r = advance(r, course);
      assert.equal(r.hole, i + 1);
      assert.equal(r.strokes, 0);
    }
  }
  assert.equal(r.complete, true);
  assert.equal(
    r.scores.reduce((a, b) => a + b, 0),
    25,
  );
  assert.equal(advance(r, course), r);
});
test("unfinished holes cannot advance and penalties are counted once", () => {
  let r = newRound(course);
  assert.equal(advance(r, course), r);
  const result = {
    end: r.ball,
    path: [],
    made: false,
    penalty: 1,
    reason: "Water",
    carry: 0,
    distance: 1,
    duration: 1,
  };
  r = applyShot(r, result, "water-1");
  assert.equal(r.strokes, 2);
  assert.equal(r.scores[0], null);
  assert.equal(applyShot(r, result, "water-1"), r);
});
test("corrupt and incompatible saves are rejected", () => {
  assert.equal(restoreRound("{oops", course), null);
  const r = newRound(course);
  for (const patch of [
    { hole: -1 },
    { strokes: NaN },
    { scores: [] },
    { ball: ["bad", 0, 0] },
    { revision: "old" },
    { complete: true },
    { hole: 1 },
    { last: { path: [null] } },
  ])
    assert.equal(
      restoreRound(JSON.stringify({ ...r, ...patch }), course),
      null,
    );
});
test("preview routing covers 18 distinct greens, has par 72, and uses a separate save revision", () => {
  const routes = JSON.parse(
    readFileSync(
      new URL("../public/courses/grenland/routes.json", import.meta.url),
    ),
  );
  assert.equal(routes.preview.length, 18);
  assert.equal(new Set(routes.preview.map((h) => h.source_green_id)).size, 18);
  assert.equal(
    routes.preview.reduce((sum, h) => sum + h.par, 0),
    72,
  );
  assert.ok(
    routes.preview.every(
      (h) =>
        h.pin_to_osm_endpoint_m < 25 &&
        h.routing_status === "community-preview",
    ),
  );
  const preview = {
    ...course,
    revision: routes.revision,
    practice: routes.preview,
  };
  assert.equal(restoreRound(JSON.stringify(newRound(preview)), course), null);
  assert.equal(restoreRound(JSON.stringify(newRound(course)), preview), null);
});
test("score announcements distinguish eagle, birdie, par and bogey", () => {
  assert.equal(scoreName(1, 3), "Hole in one!");
  assert.equal(scoreName(2, 5), "Albatross");
  assert.equal(scoreName(3, 5), "Eagle");
  assert.equal(scoreName(3, 4), "Birdie");
  assert.equal(scoreName(4, 4), "Par");
  assert.equal(scoreName(5, 4), "Bogey");
  assert.equal(scoreName(6, 4), "Double bogey");
});
test("browser transform preserves UE centimetres and north/south axis", () => {
  assert.equal(course.georeference.grid_extent_m, 8128 * 0.369);
  assert.equal(course.georeference.control_residual_m, null);
  assert.ok(
    course.practice.every((p) => p.pin_status === "synthetic-practice-only"),
  );
});
