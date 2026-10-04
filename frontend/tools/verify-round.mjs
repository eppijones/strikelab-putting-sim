import { readFileSync, writeFileSync } from "node:fs";
import {
  newRound,
  advance,
  applyShot,
  restoreRound,
  simulate,
  distance,
  bearingTo,
  surfaceAt,
  CLUBS,
} from "../src/course/engine.ts";
import { suggestShot } from "../src/course/caddie.ts";
const root = new URL("../public/courses/grenland/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", root))),
  routes = JSON.parse(readFileSync(new URL("routes.json", root)));
const course = {
  ...manifest,
  practice: routes.preview,
  revision: routes.revision,
};
function tile(spec) {
  const h = readFileSync(new URL(spec.url, root)),
    s = readFileSync(new URL(spec.surfaces, root));
  return {
    ...spec,
    heights: new Float32Array(
      h.buffer.slice(h.byteOffset, h.byteOffset + h.byteLength),
    ),
    lies: new Uint8Array(s),
  };
}
const terrain = tile(course.terrain),
  results = [];
let round = newRound(course);
for (let i = 0; i < course.practice.length; i++) {
  const hole = course.practice[i],
    world = {
      terrain,
      detail: tile(hole.tile),
      pin: hole.pin,
      stimp: 10,
      wind: [0, 0],
    };
  let lastDistance = Infinity;
  for (let attempt = 0; attempt < 15 && round.scores[i] === null; attempt++) {
    const d = distance(round.ball, hole.pin),
      lie = surfaceAt(world, round.ball[0], round.ball[2]);
    let club =
      CLUBS[
        lie === "green" && d < 35
          ? 13
          : d > 180
            ? 0
            : d > 130
              ? 5
              : d > 100
                ? 7
                : d > 70
                  ? 9
                  : d > 45
                    ? 10
                    : 12
      ];
    if (d >= lastDistance - 0.03 && club.name === "Putter") club = CLUBS[12];
    const suggestion = suggestShot(world, round.ball, club);
    const result = simulate(world, round.ball, {
      speed: (club.speed * suggestion.power) / 100,
      bearing:
        bearingTo(round.ball, hole.pin) + (suggestion.aim * Math.PI) / 180,
      launch: club.loft,
      spin: club.spin,
    });
    lastDistance = d;
    round = applyShot(round, result);
    const restored = restoreRound(JSON.stringify(round), course);
    if (!restored) throw Error("Round could not resume");
    round = restored;
  }
  results.push({
    hole: i + 1,
    name: hole.name,
    par: hole.par,
    score: round.scores[i],
    remaining_m: distance(round.ball, hole.pin),
  });
  console.log(JSON.stringify(results.at(-1)));
  if (round.scores[i] === null) break;
  round = advance(round, course);
}
const report = {
  mode: "18-hole community preview",
  method:
    "Deterministic physics from each mapped tee; caddie controls shots; save/restore after each shot. No browser or physical hardware claim.",
  complete: round.complete,
  total: round.scores.reduce((a, b) => a + (b ?? 0), 0),
  holes: results,
};
writeFileSync(
  new URL("../../docs/grenland-round-verification.json", import.meta.url),
  JSON.stringify(report, null, 2),
);
if (!report.complete) process.exitCode = 1;
