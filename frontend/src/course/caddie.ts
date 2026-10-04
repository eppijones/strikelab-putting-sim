import {
  bearingTo,
  distance,
  simulate,
  type Club,
  type Vec3,
  type World,
} from "./engine.ts";

export function suggestShot(world: World, ball: Vec3, club: Club) {
  const bearing = bearingTo(ball, world.pin);
  let best = { power: 50, aim: 0, error: Infinity };
  const evaluate = (power: number, aim: number) => {
    if (power <= 0 || power > 100) return;
    const result = simulate(world, ball, {
      speed: (club.speed * power) / 100,
      bearing: bearing + (aim * Math.PI) / 180,
      launch: club.loft,
      spin: club.spin,
    });
    const error = result.made
      ? -1
      : distance(result.end, world.pin) + result.penalty * 1000;
    if (error < best.error) best = { power, aim, error };
  };
  const limit = club.name === "Putter" ? 60 : 20;
  for (let power = 5; power <= 100; power += 5)
    for (let aim = -limit; aim <= limit; aim += 5) evaluate(power, aim);
  for (const step of [2, 0.5, 0.1]) {
    const center = { ...best };
    for (let p = -3; p <= 3; p++)
      for (let a = -3; a <= 3; a++)
        evaluate(center.power + p * step, center.aim + a * step);
  }
  return best;
}
