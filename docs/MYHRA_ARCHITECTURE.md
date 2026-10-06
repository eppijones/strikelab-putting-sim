# Myhra contracts and saves

## Units and coordinates

Local source-tile convention: +X across the tile to the right, +Y up, +Z south/down the map. Positions and velocities use metres and seconds. A bearing of zero faces −Z; positive bearing turns towards +X. Launch elevation uses degrees, spin uses rpm, and the resolved launch carries an explicit units identifier. Green speed uses labelled Stimpmeter feet as an external calibration parameter; conversion to SI occurs in the physics module.

`ShotContext.ballGround` and `pinGround` are terrain contact points. `ShotOutcome.result.path` and `end` are ball centres and include exactly one 0.02135 m radius. Scene playback never adds a second radius. Physics normalizes the starting ground elevation against the authoritative surface.

The older `SavedHole.ball` container retains its legacy Y value for compatibility: fresh tee positions historically used ground height and settled positions used sphere-centre height. Consumers treat its X/Z as authoritative and normalize its Y with `heightAt`; newly stored `ShotContext` explicitly disambiguates contact position. Old saves are preserved, not silently rewritten as a surveyed position.

## Shot transaction

`shot.ts` resolves a normalized player intent into one versioned launch. It exposes full/pitch/chip/putt launch families, absolute strength, selected putting range and visible strike assistance. Nominal preview, caddie and gameplay call the same resolver/simulator. Measured launches bypass game assistance and power mapping.

`contracts.ts` defines version-one context, intent, outcome and aliases for round/scenario records. The accepted gameplay transaction saves the resolved launch, context, compact timed path, endpoint, penalty and score together in one localStorage value before changing active-round state or playing the animation. A failed storage write rejects the shot, retains the previous round and rearms the swing. An acceptance guard rejects duplicate callbacks. The single scene impact event synchronizes the authored clip, audio and ball release; reloading does not depend on that animation completing.

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

## Retargeted actions and grip

`tools/author-golf-grip.py` fixes anatomical finger axes and opposing thumbs in Blender. `tools/retarget-mixamo-golf.py` retargets owner-downloaded Adobe Mixamo drive, chip and putt body motions to the generated avatar, correcting shoulder width, arm reach, hand size and each hand's anatomical basis. An iron action uses a separately downloaded drive motion. Grip, club-path calibration and foot plant are baked with the body. Source FBX and the new working `.blend` stay outside the repository; embedded game GLBs, the build script and trim metadata are included. The earlier `author-golf-motions.py` and editable sources remain historical fallback assets.

All clips start at zero, reach backswing at 1 s, and contact at 1.24 s. Runtime's shared impact event occurs 0.24 s after release; it releases the ball and sound once. Input scrubs the captured takeaway rather than interpolating straight between address and top. Short strokes blend their selected takeaway into the same contact and shorten the follow-through. The rendered contact check verifies the actual club-head position as well as full/partial transform parity.

AnimationMixer plays the baked action. Bounded stance correction keeps feet on the shared terrain and preserves the independent club anchor. Both hands retain their authored anatomical orientation; a two-bone arm correction and wrist residual keep the closed finger centres on the grip during animation interpolation. The previous uncorrected pose is restored before sampling so paused corrections cannot accumulate. Continuous grip checks cover all four families, three strengths and both asset tiers, including React development remounts. The seated cart pose remains procedural.

Professional golf technique is not certified by generating an action file. Coach review and final motion/contact acceptance remain release gates, with rendered three-angle evidence in `docs/rebuild/motions`.
