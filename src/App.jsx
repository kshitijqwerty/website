import {
  BrowserRouter as Router,
  Routes,
  Route,
} from "react-router-dom";

import Home from "./pages/Home";
import BlogPost from "./pages/BlogPost";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />

        <Route
          path="/blog/:slug"
          element={<BlogPost />}
        />
      </Routes>
    </Router>
  );
}