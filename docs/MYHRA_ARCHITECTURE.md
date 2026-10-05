# Myhra contracts and saves

## Units and coordinates

Local source-tile convention: +X across the tile to the right, +Y up, +Z south/down the map. Positions and velocities use metres and seconds. A bearing of zero faces −Z; positive bearing turns towards +X. Launch elevation uses degrees, spin uses rpm, and the resolved launch carries an explicit units identifier. Green speed uses labelled Stimpmeter feet as an external calibration parameter; conversion to SI occurs in the physics module.

`ShotContext.ballGround` and `pinGround` are terrain contact points. `ShotOutcome.result.path` and `end` are ball centres and include exactly one 0.02135 m radius. Scene playback never adds a second radius. Physics normalizes the starting ground elevation against the authoritative surface.

The older `SavedHole.ball` container retains its legacy Y value for compatibility: fresh tee positions historically used ground height and settled positions used sphere-centre height. Consumers treat its X/Z as authoritative and normalize its Y with `heightAt`; newly stored `ShotContext` explicitly disambiguates contact position. Old saves are preserved, not silently rewritten as a surveyed position.

## Shot transaction

`shot.ts` resolves a normalized player intent into one versioned launch. It exposes full/pitch/chip/putt launch families, absolute strength, selected putting range and visible strike assistance. Nominal preview, caddie and gameplay call the same resolver/simulator. Measured launches bypass game assistance and power mapping.

`contracts.ts` defines version-one context, intent, outcome and aliases for round/scenario records. The accepted gameplay transaction saves the resolved launch, context, compact timed path, endpoint, penalty and score together in one localStorage value before playing the animation. An acceptance guard rejects duplicate callbacks. The single scene impact event synchronizes the authored clip, audio and ball release; reloading does not depend on that animation completing.

`physicsVersion` is separate from `courseRevision`. The latter hashes authoritative height/lie bytes, pin/tee and surface-region definitions. Active saves with incompatible terrain enter explicit review; they are not automatically moved onto an updated surface. Cosmetic/quality changes do not change physics geometry or outcomes.

## Record provenance

- `strikelab.myhra.hole.v3`: accepted active-hole transaction. The previous `.v2` key is read as migration input and left untouched. Legacy summaries remain summaries, with no invented paths.
- `strikelab.myhra.settings.v1`: validated visual/input preferences.
- `strikelab.grenland.journal.v1`: entered scorecards, remembered map positions and separate what-if records. Optional information stays null/unknown. Penalties are independent events with no fabricated flight.
- Manual replay interpolates remembered endpoints along labelled illustrative display paths. It does not solve backward to invent measured launch/spin, and those display paths never enter gameplay scoring or physics statistics.
- What-if records reference the original round/shot/revision and retain an explicit changed launch/context, with a separate simulated outcome. Practice has temporary state and cannot replace an active saved round.
- Imports validate version, size, finite bounded values and record references. UI import assigns new IDs and adds copies; export keeps the versioned JSON.

Records are local to the browser origin. No account/provider integration was added. Future GPS/launch-monitor adapters can supply observations with method, approximation and revision provenance rather than masquerading as user-entered or simulated measurements.

## Surface and rendering

`heightAt` consumes the authoritative source grid and the shared detail patch. `terrain-basins.f32` / `green-basins.f32` are authored bowl/lip derivatives; the original source grids remain available. The putting grid and preview sample the same heights as collision and the displayed playing corridor. The entire Myhra corridor keeps its authoritative grid triangles across every visual tier; only distant non-playing display geometry, textures, foliage, shadows, reflections and pixel ratio are reduced.

Near trees use 3D meshes; far vegetation uses the existing atlas. Mobile GLBs and the smaller HDR are visual derivatives. They do not alter tree collision footprints or playing surfaces. Cart simulation uses a 1/120 s step, bounded acceleration, steering and boost, with body-footprint surface/tree collisions.

## Authored actions

`tools/author-golf-grip.py` fixes anatomical finger axes and opposing thumbs in Blender. `tools/author-golf-motions.py` bakes `GolfDriver`, `GolfIron`, `GolfChip`, `GolfPutt` and the club anchor into GLB actions. Editable `.blend` sources are under `tools/art`. Runtime procedural work is limited to stance/terrain presentation and the existing seated-cart pose; shot actions are played by AnimationMixer.

Professional golf technique is not certified by generating an action file. Coach review and final motion/contact acceptance remain release gates, with rendered three-angle evidence in `docs/rebuild/motions`.
