import {useCallback, useEffect, useRef, useState} from 'react';
import {SwingController, type Strike, type SwingMode, type SwingSource} from './swing';

export function useSwing(onStrike: (strike: Strike) => void, enabled: boolean, mode: SwingMode) {
  const controller = useRef(new SwingController());
  const callback = useRef(onStrike), allowed = useRef(enabled);
  const [view, setView] = useState(() => new SwingController().view());
  useEffect(() => {
    callback.current = onStrike; allowed.current = enabled;
    if (!enabled && controller.current.phase !== 'finish') controller.current.cancel();
  }, [onStrike, enabled]);
  useEffect(() => {
    if (controller.current.phase === 'finish') controller.current.mode = mode;
    else controller.current.reset(mode);
    setView(controller.current.view());
  }, [mode]);
  useEffect(() => {
    const activeController = controller.current;
    let raf = 0, last = 0;
    const frame = (now: number) => {
      controller.current.update(now);
      if (now - last >= 16) {
        const next = controller.current.view();
        setView(previous => JSON.stringify(previous) === JSON.stringify(next) ? previous : next);
        last = now;
      }
      raf = requestAnimationFrame(frame);
    };
    const cancel = () => { if (controller.current.phase !== 'finish') controller.current.cancel(); };
    const hidden = () => { if (document.hidden) cancel(); };
    const orientation = () => cancel();
    const disconnect = () => { if (controller.current.source === 'controller') cancel(); };
    addEventListener('blur', cancel); addEventListener('pagehide', cancel);
    addEventListener('visibilitychange', hidden); addEventListener('orientationchange', orientation);
    addEventListener('gamepaddisconnected', disconnect);
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf); activeController.cancel();
      removeEventListener('blur', cancel); removeEventListener('pagehide', cancel);
      removeEventListener('visibilitychange', hidden); removeEventListener('orientationchange', orientation);
      removeEventListener('gamepaddisconnected', disconnect);
    };
  }, []);
  const emit = useCallback((strike: Strike | undefined) => { if (strike && allowed.current) callback.current(strike); }, []);
  const begin = useCallback((source: SwingSource = 'mouse') => { if (allowed.current) controller.current.begin(performance.now(), source); }, []);
  const move = useCallback((pull: number, side: number) => { if (allowed.current) emit(controller.current.move(pull, side, performance.now())); }, [emit]);
  const release = useCallback(() => { if (allowed.current) emit(controller.current.release(performance.now())); }, [emit]);
  const press = useCallback(() => { if (allowed.current) emit(controller.current.press(performance.now())); }, [emit]);
  const cancel = useCallback(() => controller.current.cancel(), []);
  const reset = useCallback(() => controller.current.reset(), []);
  const controllerMove = useCallback((pull: number, side: number, connected = true) => {
    if (allowed.current) emit(controller.current.controllerMove(pull, side, performance.now(), connected));
    else if (controller.current.phase !== 'finish') controller.current.cancel();
  }, [emit]);
  return {view, controller, begin, move, release, press, reset, cancel, controllerMove};
}
