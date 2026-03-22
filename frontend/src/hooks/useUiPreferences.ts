import { useEffect, useState } from 'react';

const SHOW_DISTANCE_KEY = 'strikelab.showDistance';

export function useUiPreferences() {
  const [showDistance, setShowDistance] = useState<boolean>(() => {
    const stored = window.localStorage.getItem(SHOW_DISTANCE_KEY);
    return stored === null ? true : stored === 'true';
  });

  useEffect(() => {
    window.localStorage.setItem(SHOW_DISTANCE_KEY, String(showDistance));
  }, [showDistance]);

  return {
    showDistance,
    setShowDistance,
  };
}
