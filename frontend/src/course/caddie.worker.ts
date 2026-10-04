import { suggestShot } from "./caddie";
import type { Club, Vec3, World } from "./engine";

self.onmessage = (
  event: MessageEvent<{ world: World; ball: Vec3; club: Club }>,
) => {
  const { world, ball, club } = event.data;
  self.postMessage(suggestShot(world, ball, club));
};
