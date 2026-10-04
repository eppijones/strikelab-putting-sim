import { StrictMode, lazy, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
const App = lazy(() => import('./App.tsx'))

const CourseApp = lazy(() => import('./course/CourseApp'));

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Suspense fallback={<div>Loading StrikeLab…</div>}>{location.pathname.startsWith('/play/grenland') ? <CourseApp /> : <App />}</Suspense>
  </StrictMode>,
)

if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => { navigator.serviceWorker.register("/sw.js").catch(() => { /* Online play remains available. */ }); });
}
