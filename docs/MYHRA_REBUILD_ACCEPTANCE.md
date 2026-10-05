# Myhra rebuild — review preview

6 October 2026 (work began 5 October). Branch `codex/grenland-finished-hole`, based on `128fdff`. Production remains unchanged until release acceptance passes. This is a working implementation for review; **the complete finished-quality milestone has not yet passed**.

## What is implemented

- Touch: fixed 96 CSS-pixel travel, live absolute power, release to strike, side cancellation. Mouse/controller: deliberate forward crossing; neutral return does not fire. Three-click: start, power, accuracy.
- Shared fixed-step physics for nominal preview, caddie and delivered shots. Full flight/landing/roll preview, terrain-following putting grid, downhill flow, break path, centimetre elevation, labelled 6/15/35 m ranges. Visible forgiving strike setting; measured launches bypass it.
- Blender-authored grip and four baked actions: driver, iron, chip and putt. Partial swings converge on the same square contact pose, then shorten the follow-through. One scene impact event releases the ball and triggers sound. These are authored animations, not professional mocap; coach approval is pending.
- One camera owner, independent aim/orbit input, stable short-putt framing, deliberate long-putt tracking, overview, cart and illustrative replay.
- Compact caddie; contextual controls; mobile sheets and 44 CSS-pixel primary targets; persistent preferences and E/+n/−n hole scoring.
- Separate practice state; atomic accepted-shot/score saves; retained original legacy key; course-revision review instead of silently moving a saved ball.
- Manual 18-hole scorecard, optional historical setup, map endpoints and independent penalties, labelled approximate replay, play/pause/scrub, separate what-if outcomes, versioned JSON export/import. A 77 benchmark contains no invented hole scores or trajectories.
- Shared bunker bowl/lip heights, nearby 3D firs, distant atlas vegetation, readable pond shoreline/water, daylight controls, clear cart windscreen, fixed-step steering/braking/reverse/collisions and bounded boost.
- Lower-resolution mobile visual derivatives; authoritative playing geometry remains identical across tiers. Far scenery is streamed separately from the critical playable assets.

## Acceptance evidence

| Gate | Status | Evidence and limits |
| --- | --- | --- |
| TypeScript and production build | PASS | Vite build; expected large Three.js chunk warning. |
| Frontend test suite | PASS | 66 tests: swing, physics, cup/surfaces/penalties, saves, journal, contracts and cart. |
| Changed production-source lint | PASS | Scoped ESLint. Repository-wide lint still has unrelated existing lab errors. |
| Tee → approach → putt → score → reload | PASS, automated | Real UI/caddie/touch loop; three strokes in the fixture. `rebuild/golf-loop.json`. This is completion evidence, not realistic scoring difficulty. |
| Default touch, cancellation, rotation, practice and cart | PASS, emulated | `rebuild/browser-flow.json`; Chrome with a phone viewport and CDP touch input. |
| Secondary touch, interruption and controller reconnect | PASS, automated/injected | `rebuild/input-robustness.json`; synthetic blur/orientation and standard-gamepad injection. Actual mobile backgrounding and physical disconnect remain untested. |
| Three-click and journal export/import | PASS, automated | Input report; no stroke before the third click; import preserves originals as separate copies. |
| WebKit load/practice/resume | PASS, emulated | Desktop WebKit with phone viewport. This is not a physical Safari/iPhone result. |
| 30/60/120 Hz input | PASS, automated | Equivalent sampled normalized strokes within one percentage point power and 0.1° direction. Fixed-step cart travel also agrees. |
| Prediction/play parity | PASS, automated | Perfect delivered launch and nominal preview produce identical outcomes, including cup/penalty results. |
| Carry/apex and flat-putt calibration | PARTIAL | Published Trackman aggregate fixtures meet 5% carry / 10% apex; USGA standard-roll fixture and analytic slope/flat-roll checks pass. Independent measured bounce, surface rollout and cup-entry fixtures are still missing. |
| Rig/motion visual review | PARTIAL | Four families × front/side/rear × four poses plus slow-motion footage in `rebuild/motions/`. Same-frame impact is implemented, but professional technique and terrain-contact acceptance need coach/player review. |
| Partial-swing contact | PASS, rendered check | Driver/iron/chip/putt at 100%, 45% and 8% reach the identical authored grip/club impact transform. `rebuild/impact/checks.json` and partial-stroke footage. This does not certify professional technique. |
| Historical replay integrity | PASS, automated | Penalties create no flight, originals remain unchanged, scenario/journal reload and JSON copies preserve data. |
| Cold playable load | PASS, emulated sample | 7.117 s at 20 Mbps / 80 ms, cold cache with service workers blocked; 7,920,973 bytes at readiness. `rebuild/load-measurement.json`. Local preview, not a measured physical device or Vercel CDN result. |
| Mobile working geometry budget | PASS, sampled tee | 333k visible triangles, 29 draw calls on balanced tier. Frame time, scenery position and real-device results govern final acceptance. |
| Texture residency | UNMEASURED | Approximate cold balanced-tier allocation around 126 MB; no physical GPU residency trace. Tier changes can retain previously loaded textures. |
| 20-minute mixed-play browser session | See `rebuild/soak.json` | Putting, driving, replay, preserved unfinished round and runtime errors. Browser RAF timings on the available PC are not physical mobile acceptance. |
| Visual authenticity and moving-camera approval | PENDING | Tee/corridor references considered; 2024 green/pond changes documented. Exact boundaries, pond dimensions and green contours are not surveyed or club-approved. |
| iPhone 15 Pro / iPhone 11 | UNTESTED | Physical portrait/landscape frame-time, thermal behaviour, input latency and OS backgrounding. |
| Actual DualSense | UNTESTED | USB/Bluetooth controls, disconnect/reconnect and stick feel. Injected events do not count. |
| i5-10400 / GTX 1660 reference PC | UNTESTED | Available-machine browser automation does not establish this hardware result. |
| Input response ≤75 ms / settled prediction ≤100 ms | UNMEASURED | Functional checks pass; no end-to-end instrumented latency measurement yet. |

## Course and animation provenance

The [club's 2024 report](https://grenlandgolf.no/wp-content/uploads/2025/03/Grenland-og-Omegn-Golfklubb-Arsberetning-2024-17-03-25.pdf) documents Hole 1 green/foregreen work and a pond. It does not supply surveyed dimensions. Source terrain predates these changes; bunker basins/lips and detailed green geometry remain explicitly authored approximations. Render, collision, reading grid and ball prediction consume the same chosen height surface.

Poly Haven assets permit redistribution under [CC0](https://polyhaven.com/license). Existing generated golfer/cart/birch assets come from the owner's Tripo account; the visible account showed Pro, and [paid-user commercial-use guidance](https://www.tripo3d.ai/help/privacy-policy/how-to-use-tripo-models-commercially) was reviewed. `frontend/public/courses/grenland/myhra/sources.json` records transformations and the rights-review evidence. New credit spend: **0**.

## Before promotion

1. Review tee, approach, bunker, pond/green and putting in motion with a Grenland golfer; correct remaining reference mismatches and any distracting foliage transitions.
2. Have a golf coach review all four actions and impact/grip/feet from three angles. Replace or improve any motion that fails that review.
3. Complete independent bounce/roll/cup fixtures and instrument the input/prediction latency gates.
4. Run the physical device/controller matrix and a controlled 20-minute session on each required tier. Record p95/p99, thermal behaviour, real cold load and resident textures.
5. Promote only after these gates pass. Clubhouse, range, connected cart travel and remaining holes follow the Myhra milestone.

## Reproduce

From `frontend`, run `npm test`, `npm run build`, and scoped ESLint on changed production sources. Serve the production build at port 4175. Run `node tools/verify-rebuild.mjs`, `node tools/verify-golf-loop.mjs`, `node tools/verify-input-rebuild.mjs`, `node tools/measure-rebuild.mjs`, and `node tools/soak-rebuild.mjs`. Set `GRENLAND_TEST_URL` to a reachable preview for flow checks. Motion review uses the dev-only asset tool on port 5175.

The original checkout, unrelated local changes and production deployment are preserved. A Vercel preview uses a different origin, so browser-local production saves will not appear there automatically; journal JSON can be exported and imported as copies.

Before/after browser footage is in `rebuild/comparison/index.html`: the baseline is a production build from a Git archive of `128fdff`; both scenes use a fresh phone-sized viewport. The original motion footage covers full strokes, while `rebuild/impact/partial-strokes.webm` captures the final partial-swing correction.
