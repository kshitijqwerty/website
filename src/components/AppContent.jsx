import { lazy, Suspense } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";

import AnimatedPage from "./AnimatedPage";

const Home = lazy(() => import("../pages/Home"));
const BlogIndex = lazy(() => import("../pages/BlogIndex"));
const BlogPost = lazy(() => import("../pages/BlogPost"));

function NotFound() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex items-center justify-center px-6">
      <div className="text-center">
        <h1 className="text-6xl font-bold font-heading mb-4">404</h1>
        <p className="text-neutral-400 mb-8">Page not found.</p>
        <a
          href="/"
          className="btn-primary bg-white text-black px-6 py-3 rounded-2xl font-medium hover:opacity-90 transition-opacity inline-block"
        >
          Go Home
        </a>
      </div>
    </div>
  );
}

export default function AppContent() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<AnimatedPage><Home /></AnimatedPage>} />
        <Route path="/blog" element={<AnimatedPage><BlogIndex /></AnimatedPage>} />
        <Route path="/blog/:slug" element={<AnimatedPage><BlogPost /></AnimatedPage>} />
        <Route path="*" element={<AnimatedPage><NotFound /></AnimatedPage>} />
      </Routes>
    </AnimatePresence>
  );
}
