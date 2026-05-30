import { lazy, Suspense } from "react";
import { BrowserRouter as Router } from "react-router-dom";

import ErrorBoundary from "./components/ErrorBoundary";
import Navbar from "./components/Navbar";
import BackToTop from "./components/BackToTop";

const AppContent = lazy(() => import("./components/AppContent"));

export default function App() {
  return (
    <Router>
      <ErrorBoundary>
        <Navbar />
        <Suspense fallback={null}>
          <AppContent />
        </Suspense>
        <BackToTop />
      </ErrorBoundary>
    </Router>
  );
}
