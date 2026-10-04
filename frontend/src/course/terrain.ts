import { BufferAttribute, BufferGeometry } from "three";
import { contains, type Tile, type TileSpec } from "./engine";

export const COURSE_ROOT = "/courses/grenland/";
export async function loadTile(
  spec: TileSpec,
  signal?: AbortSignal,
): Promise<Tile> {
  const [heights, lies] = await Promise.all(
    [spec.url, spec.surfaces].map(async (url) => {
      const response = await fetch(COURSE_ROOT + url, { signal });
      if (!response.ok)
        throw new Error(`Terrain download failed (${response.status})`);
      return response.arrayBuffer();
    }),
  );
  if (
    heights.byteLength !== spec.size * spec.size * 4 ||
    lies.byteLength !== spec.size * spec.size
  )
    throw new Error("Terrain file dimensions do not match the course manifest");
  return {
    ...spec,
    heights: new Float32Array(heights),
    lies: new Uint8Array(lies),
  };
}
const colors = [
  [0.22, 0.34, 0.16],
  [0.32, 0.46, 0.23],
  [0.43, 0.58, 0.29],
  [0.75, 0.68, 0.48],
];
export function terrainGeometry(
  tile: Tile,
  hole?: TileSpec,
  extent = (tile.size - 1) * tile.spacing,
) {
  const n = tile.size,
    positions = new Float32Array(n * n * 3),
    color = new Float32Array(n * n * 3),
    uvs = new Float32Array(n * n * 2),
    indices: number[] = [];
  for (let z = 0; z < n; z++)
    for (let x = 0; x < n; x++) {
      const i = z * n + x,
        wx = tile.origin[0] + x * tile.spacing,
        wz = tile.origin[1] + z * tile.spacing;
      positions.set([wx, tile.heights[i], wz], i * 3);
      uvs.set([wx / extent, 1 - wz / extent], i * 2);
      const c = colors[tile.lies[i]] ?? colors[0];
      const noise = 1 + Math.sin(x * 2.13 + z * 3.17) * 0.025;
      color.set(
        c.map((v) => v * noise),
        i * 3,
      );
      if (
        x < n - 1 &&
        z < n - 1 &&
        !(
          hole &&
          contains(hole, wx, wz) && contains(hole, wx+tile.spacing, wz+tile.spacing)
        )
      )
        indices.push(i, i + n, i + 1, i + 1, i + n, i + n + 1);
    }
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(positions, 3));
  geometry.setAttribute("color", new BufferAttribute(color, 3));
  geometry.setAttribute("uv", new BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}
