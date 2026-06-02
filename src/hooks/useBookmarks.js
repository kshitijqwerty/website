import { useState, useCallback, useRef } from "react";

function getCookie(name) {
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`));
  return match ? decodeURIComponent(match[1]) : null;
}

function setCookie(name, value, days = 365) {
  const expires = new Date(Date.now() + days * 864e5).toUTCString();
  document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/; SameSite=Lax`;
}

export function useBookmarks(key = "default") {
  const cookieName = useRef(`${key}-bookmarks`).current;
  const [bookmarks, setBookmarks] = useState(() => {
    try {
      const raw = getCookie(cookieName);
      return raw ? new Set(JSON.parse(raw)) : new Set();
    } catch {
      return new Set();
    }
  });

  const toggle = useCallback((slug) => {
    setBookmarks((prev) => {
      const next = new Set(prev);
      if (next.has(slug)) {
        next.delete(slug);
      } else {
        next.add(slug);
      }
      setCookie(cookieName, JSON.stringify([...next]));
      return next;
    });
  }, []);

  const isBookmarked = useCallback(
    (slug) => bookmarks.has(slug),
    [bookmarks]
  );

  return { bookmarks, toggle, isBookmarked };
}
