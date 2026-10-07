# Myhra rebuild — review preview

7 October 2026 release check (implementation and final automated session completed 6 October; work began 5 October). Branch `codex/grenland-finished-hole`, based on `128fdff`. Gameplay code commit: `17df1398bcd8c6a96ea0ac6ff4ada8c8b59b001e`; reviewed production build cache: `d5c8a196ac7178d4`. Production remains unchanged until release acceptance passes. This is a working implementation for review; **the complete finished-quality milestone has not yet passed**.

[Play the deployed preview](https://grenland-golf-qqyxh3wj0-strikelabs-projects-8daa3b0e.vercel.app/play/grenland/myhra). Vercel's deployment dashboard reports **Ready** for `17df139`. The hosted game has not been inspected: access was initially declined; the owner subsequently authorized verification on 7 October, but automatic browser approval still rejected it because a saved browser permission blocks this domain. Local production-build results below remain valid; they do not substitute for a hosted smoke check. Vercel's existing preview sign-in protection remains enabled.

## What is implemented

- Touch: fixed 96 CSS-pixel travel, live absolute power, release to strike, side cancellation. Mouse/controller: deliberate forward crossing; neutral return does not fire. Three-click: start, power, accuracy.
- Shared fixed-step physics for nominal preview, caddie and delivered shots. Full flight/landing/roll preview, terrain-following putting grid, downhill flow, break path, centimetre elevation, labelled 6/15/35 m ranges. Visible forgiving strike setting; measured launches bypass it.
- Owner-downloaded Mixamo golf body motions retargeted in Blender for driver, iron, chip and putt, with corrected shoulder/arm proportions, hand size and anatomical grip orientation. Club-path and grip calibration accompany the motion. Partial swings converge on square contact, then shorten the follow-through. One impact event releases ball and sound. Mixamo sourcing does not certify professional technique; coach approval is pending.
- One camera owner, independent aim/orbit input, stable short-putt framing, deliberate long-putt tracking, overview, cart and illustrative replay.
- Caddie reduced to one thumb-sized recommendation row, with details on demand; contextual controls; mobile sheets and 44 CSS-pixel primary targets; persistent preferences and E/+n/−n hole scoring.
- Reload selects an appropriate approach club or putter and short-putt strength. Active wind survives reload; temporary practice/what-if wind does not overwrite the saved-round preference.
- Separate practice state; atomic accepted-shot/score saves; storage-full rejection before scoring/animation with usable retry; retained original legacy key; course-revision review instead of silently moving a saved ball.
- Manual 18-hole scorecard, optional historical setup, map endpoints and independent penalties, labelled approximate replay, play/pause/scrub, separate what-if outcomes, versioned JSON export/import. A 77 benchmark contains no invented hole scores or trajectories.
- Shared bunker bowl/lip heights, nearby 3D firs, distant atlas vegetation, readable pond shoreline/water, daylight controls, clear cart windscreen, fixed-step steering/braking/reverse/collisions and bounded boost.
- Lower-resolution mobile visual derivatives; authoritative playing geometry remains identical across tiers. Far scenery is streamed separately from the critical playable assets.

## Acceptance evidence

| Gate | Status | Evidence and limits |
| --- | --- | --- |
| TypeScript and production build | PASS | Vite build; expected large Three.js chunk warning. |
| Vercel preview build | PASS, deployment status | Dashboard reports Ready for gameplay commit `17df139`. Production remains at `128fdff`. |
| Hosted gameplay smoke check | UNTESTED, saved browser permission blocked | The owner authorized verification, but the browser still rejects this domain due to a saved permission setting. Local production-build gameplay has been checked; CDN delivery and signed-in hosted interaction have not. |
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
| Partial-swing contact | PASS, rendered check | Driver/iron/chip/putt at 100%, 45% and 8% reach the identical grip/club impact transform; the actual head is within 1 cm of its calibrated contact position. `rebuild/impact/checks.json` and partial-stroke footage. This does not certify professional technique. |
| Continuous grip and hand orientation | PASS, rendered bone check | Driver/iron/chip/putt at 100%, 45% and 8%, throughout takeaway/downstroke/follow-through on desktop and mobile assets. Authored closed finger centres stay on the shaft and the index/pinky orientation does not invert. React development remounts are included. `rebuild/grip/checks.json`. Finger mesh deformation and professional technique still require visual approval. |
| Terrain stance | PASS, rendered bone check | Four families on flat and ±30% authored planes: foot contact, grip preservation and paused-pose drift. `rebuild/ground-contact/checks.json`. Final shoe deformation and professional technique remain visual review gates. |
| Bunker, water and daylight views | PASS, automated flow / review evidence | `rebuild/views/review.json`: bunker shot, water penalty/relief/reload, putting and moving views. Morning/midday/evening footage is review material, not surveyed-course approval. |
| Historical replay integrity | PASS, automated | Penalties create no flight, originals remain unchanged, scenario/journal reload and JSON copies preserve data. |
| Cold playable load | PASS, emulated sample | 7.305 s at 20 Mbps / 80 ms, cold cache with service workers blocked; 9,673,722 bytes at readiness. `rebuild/load-measurement.json`. Local preview, not a measured physical device or Vercel CDN result. |
| Mobile working geometry budget | PASS, sampled tee | 333k visible triangles, 29 draw calls on balanced tier. Frame time, scenery position and real-device results govern final acceptance. |
| Texture residency | UNMEASURED | Approximate cold balanced-tier allocation around 126 MB; no physical GPU residency trace. Tier changes can retain previously loaded textures. |
| 20-minute mixed-play browser session | PASS, automated | 1,210.5 seconds, 69 activities, no runtime errors and no changes to the unfinished active round during separate practice/replay. Browser RAF p95 8.4 ms / p99 8.5 ms, 140,985 frames. `rebuild/soak.json` identifies the tested final bundle. Available-PC headless timings are not physical mobile acceptance. |
| Visual authenticity and moving-camera approval | PENDING | Tee/corridor references considered; 2024 green/pond changes documented. Exact boundaries, pond dimensions and green contours are not surveyed or club-approved. |
| iPhone 15 Pro / iPhone 11 | UNTESTED | Physical portrait/landscape frame-time, thermal behaviour, input latency and OS backgrounding. |
| Actual DualSense | UNTESTED | USB/Bluetooth controls, disconnect/reconnect and stick feel. Injected events do not count. |
| i5-10400 / GTX 1660 reference PC | UNTESTED | Available-machine browser automation does not establish this hardware result. |
| Input response ≤75 ms / settled prediction ≤100 ms | PARTIAL, browser measurement | DOM event → rendered value p95 12.6 ms (max 13.0); projected-distance settling p95 24.6 ms (max 30.7), 20 samples. `rebuild/input-latency.json`. Physical touch glass, controller, display scanout and thermal behaviour remain untested. |

## Course and animation provenance

The [club's 2024 report](https://grenlandgolf.no/wp-content/uploads/2025/03/Grenland-og-Omegn-Golfklubb-Arsberetning-2024-17-03-25.pdf) documents Hole 1 green/foregreen work and a pond. It does not supply surveyed dimensions. Source terrain predates these changes; bunker basins/lips and detailed green geometry remain explicitly authored approximations. Render, collision, reading grid and ball prediction consume the same chosen height surface.

Poly Haven assets permit redistribution under [CC0](https://polyhaven.com/license). Existing generated golfer/cart/birch assets come from the owner's Tripo account; the visible account showed Pro, and [paid-user commercial-use guidance](https://www.tripo3d.ai/help/privacy-policy/how-to-use-tripo-models-commercially) was reviewed. [Adobe's Mixamo FAQ](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html) permits video-game use. Raw FBX and the new working Blender file remain local outside the repository. `frontend/public/courses/grenland/myhra/sources.json` records transformations and rights evidence. New credit spend: **0**.

## Before promotion

1. Review tee, approach, bunker, pond/green and putting in motion with a Grenland golfer; correct remaining reference mismatches and any distracting foliage transitions.
2. Have a golf coach review all four actions and impact/grip/feet from three angles. Replace or improve any motion that fails that review.
3. Complete independent bounce/roll/cup fixtures and physical input/prediction latency measurements.
4. Run the physical device/controller matrix and a controlled 20-minute session on each required tier. Record p95/p99, thermal behaviour, real cold load and resident textures.
5. Promote only after these gates pass. Clubhouse, range, connected cart travel and remaining holes follow the Myhra milestone.

## Reproduce

From `frontend`, run `npm test`, `npm run build`, and scoped ESLint on changed production sources. Serve the production build at port 4175. Run `node tools/verify-rebuild.mjs`, `node tools/verify-golf-loop.mjs`, `node tools/verify-input-rebuild.mjs`, `node tools/measure-rebuild.mjs`, and `node tools/soak-rebuild.mjs`. Set `GRENLAND_TEST_URL` to a reachable preview for flow checks. Motion review uses the dev-only asset tool on port 5175.

The original checkout, unrelated local changes and production deployment are preserved. A Vercel preview uses a different origin, so browser-local production saves will not appear there automatically; journal JSON can be exported and imported as copies. Vercel's existing preview protection remains enabled: signed-in Chrome can open it, while an unauthenticated isolated browser reaches the Vercel login page. Public release remains gated by the acceptance matrix.

Before/after browser footage is in `rebuild/comparison/index.html`: the baseline is a production build from a Git archive of `128fdff`; both scenes use a fresh phone-sized viewport. The full-action footage includes the final captured takeaway and runtime grip correction; `rebuild/impact/partial-strokes.webm` covers the three tested strength levels.
