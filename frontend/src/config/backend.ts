export const BACKEND_HTTP_URL = import.meta.env.VITE_BACKEND_HTTP_URL ?? 'http://localhost:8000';
export const BACKEND_WS_URL = import.meta.env.VITE_BACKEND_WS_URL ?? 'ws://localhost:8000/ws';

export function apiUrl(path: string): string {
  return `${BACKEND_HTTP_URL}${path}`;
}
