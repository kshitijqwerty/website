import {
  BrowserRouter as Router,
  Routes,
  Route,
} from "react-router-dom";

import ErrorBoundary from "./components/ErrorBoundary";
import Home from "./pages/Home";
import BlogPost from "./pages/BlogPost";

export default function App() {
  return (
    <Router>
      <ErrorBoundary>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/blog/:slug" element={<BlogPost />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </ErrorBoundary>
    </Router>
  );
}

function NotFound() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex items-center justify-center px-6">
      <div className="text-center">
        <h1 className="text-6xl font-bold font-heading mb-4">404</h1>
        <p className="text-neutral-400 mb-8">Page not found.</p>
        <a
          href="/"
          className="px-6 py-3 rounded-2xl bg-white text-black font-medium hover:opacity-90 transition-opacity inline-block"
        >
          Go Home
        </a>
      </div>
    </div>
  );
}