import { useEffect, useRef, useState } from 'react';

import { BACKEND_HTTP_URL, apiUrl } from '../config/backend';

export function useBackendHealth() {
  const [backendReady, setBackendReady] = useState(false);
  const timeoutRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    let delay = 1000;

    const poll = async () => {
      try {
        const response = await fetch(apiUrl('/api/health'));
        if (response.ok) {
          const data = await response.json();
          if (data.status === 'ready' || data.status === 'degraded') {
            if (!cancelled) {
              setBackendReady(true);
            }
            return;
          }
        }
      } catch {
        // Backend not up yet.
      }

      if (!cancelled) {
        delay = Math.min(delay * 1.3, 4000);
        timeoutRef.current = window.setTimeout(poll, delay);
      }
    };

    poll();
    return () => {
      cancelled = true;
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return {
    backendReady,
    httpBaseUrl: BACKEND_HTTP_URL,
  };
}
