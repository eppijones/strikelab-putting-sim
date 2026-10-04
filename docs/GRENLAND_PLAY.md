# Grenland Golf — browser release

Updated 4 October 2026. Public game: https://grenland-golf.vercel.app/play/grenland

The Vercel project `grenland-golf` builds `frontend` from `eppijones/strikelab-putting-sim`, branch `master`. The public root redirects to the game. No account or simulator is needed to play. The existing StrikeLab marketing site is separate.

## Play

Choose a club, aim, and target power. Pull down on the swing pad and smoothly return to the starting point. Backswing sets the fraction of target power; swing path and downswing tempo affect direction and contact. The preview distance assumes a clean strike.

| Input | Controls |
| --- | --- |
| Mouse / touch | Drag down then up on the swing pad. Drag the course to orbit and pinch/scroll to zoom. Sliders set aim and target power. |
| Keyboard | Space starts the meter, then three timed presses set power, path and tempo. Arrows adjust aim/power. C changes club; V scouts; G toggles the green grid; N advances after holing; Escape closes dialogs. |
| Standard gamepad / PS5 layout | Right stick backswing/downswing; left stick aim/power; L1/R1 club; X timing swing; Square scout; Circle next/close; Triangle play/walk/cart; Options scorecard. |
| Walk / cart | WASD, left stick or on-screen direction buttons. A gold marker identifies the next ball/tee. Return to ball resumes golf. After holing, choose Next tee or Walk to next tee. |

Settings include three-click touch controls, two graphics levels, wind, green speed, sound, 18-hole routing or 25 practice greens, and practice restart positions. Each layout has an independent local save. Round score appears as E / +1 / −2 and the scorecard records each completed hole. Saves belong to this browser and origin; they do not sync across devices. Importing the historical 77 round is deferred by agreement.

On iPhone, open the HTTPS link in Safari or Chrome. Add to Home Screen is optional. The course can reload offline after the initial download completes. A first visit requires internet access. The next reload uses an installed update, while the previous build remains cached for an already-open page.

## Local launch and simulator

Double-click `Play Grenland.cmd` in the repository. It serves the built game at http://127.0.0.1:8088/play/grenland and prints a LAN address. Python is required for this launcher. To rebuild, run `npm --prefix frontend run build`.

The optional physical simulator uses the existing backend's `/ws/shots` stream. Calibrate the measurement setup first, then enter its WebSocket address in Settings. Frozen launch measurements are journaled in SQLite, identified by device and shot, delivered with resume cursors, and acknowledged after the game saves the outcome. Reconnect does not apply a committed shot twice. Camera images are not sent to Vercel.

For local HTTP play, use `ws://SIMULATOR-LAN-ADDRESS:8000/ws/shots`. The public HTTPS game requires a trusted `wss://` endpoint for the simulator. A remote relay, physical installation and TLS endpoint have not been provisioned. Full-swing/launch/spin measurements require compatible hardware; the current camera integration supplies measured putts. Synthetic bridge tests do not establish physical calibration.

## Implementation

- Native browser WebGL 2 renderer with an animated third-person golfer, walk/cart modes, ball-follow and scout cameras, shadows, slope grid and trajectory preview.
- Existing MetaHuman body/clothing exported through FBX and Blender to repair Unreal GLB skin weights. Existing Megaplant Norway maple crown rebuilt with explicit textured leaf cards. Existing DZ_Grasses ground materials replace the satellite image as the playing surface. Asset provenance is in `frontend/public/courses/grenland/art/sources.json`.
- Real source terrain and detailed green tiles; 14 clubs; fixed-step flight, bounce and roll; surface friction, cup capture, wind and penalties; local caddie calculations in a worker.
- No Spiderbench code or assets incorporated. No Tripo/fal generation credits or paid add-ons used.

## Verification

- Production TypeScript/Vite build and course ESLint pass. 26 engine and swing tests cover impact timing, power/path/contact, cancellation, exactly-once strikes, cup capture, surface friction, slope, penalties, saves and routing.
- Chrome and WebKit automated phone-layout checks cover loading, timed putting, scoring, reload, next-hole walking, cart movement, scouting and landscape. No JavaScript errors or failed same-origin requests in those runs. Evidence: `grenland-chromium-verification.json` and `grenland-webkit-verification.json`.
- Mocked standard gamepad and real browser mouse events verify a full right-stick shot, idle-controller/mouse coexistence, one strike per gesture, and offline round recovery. Evidence: `grenland-input-verification.json`.
- `node frontend/tools/verify-round.mjs` completes all 18 mapped holes through the real physics and caddie, restoring after every shot: 49 strokes. This tests completion, not realistic scoring difficulty; the caddie has exact knowledge of the simulation.
- Backend suite from the integration pass: 32 passed, 3 video-dependent tests skipped. Frontend dependency audit after compatible updates: zero reported vulnerabilities.
- Physical iPhone performance/thermal behavior, Bluetooth/USB DualSense behavior and installed simulator accuracy still require real-device acceptance testing. Browser emulation is not a substitute for those measurements.

Reproduce browser checks with `node frontend/tools/browser-smoke.mjs chromium URL`, `node frontend/tools/browser-smoke.mjs webkit URL`, and `node frontend/tools/input-smoke.mjs URL`. Install the WebKit test engine with `npx --prefix frontend playwright install webkit` if needed.

## Course authenticity

This remains a community-routed Grenland preview. The 18 tees follow OpenStreetMap routes; pins are synthetic interior positions in the existing green data. Club-approved tees/pins, current hazards, paths/buildings and surveyed accuracy have not been established. The preview is par 72. The older club guide shown separately sums to par 71. Routing provenance and raw ODbL data are included in `routes.json` and `routing-osm.json`.

Terrain comes from the existing Kartverket-derived source. Golf flight/roll are game physics, not a calibrated measurement instrument. The browser assets are optimized versions of the owned project assets, not a claim of visual parity with PGA Tour or the native Unreal renderer.
