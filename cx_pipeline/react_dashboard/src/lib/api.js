import { useCallback, useEffect, useRef, useState } from "react";

/* Every superseded request is aborted, and responses are memoised by URL so
   back/forward and re-selecting a filter are instant instead of a round trip. */

const CACHE_TTL_MS = 30_000;
const CACHE_MAX = 120;
const cache = new Map();

function readCache(url) {
  const hit = cache.get(url);
  if (!hit) return undefined;
  if (Date.now() - hit.at > CACHE_TTL_MS) {
    cache.delete(url);
    return undefined;
  }
  return hit.data;
}

function writeCache(url, data) {
  cache.set(url, { at: Date.now(), data });
  if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
}

export function clearCache() {
  cache.clear();
}

export async function apiGet(url, { signal } = {}) {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

/** apiGet that reads and fills the shared response cache. */
export async function apiGetCached(url, { signal } = {}) {
  const hit = readCache(url);
  if (hit !== undefined) return hit;
  const data = await apiGet(url, { signal });
  writeCache(url, data);
  return data;
}

export async function apiPost(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetch `url` whenever it changes.
 * `url` of null means "nothing to load" — resolves to null without a request.
 * Keeps the previous data visible while the next request is in flight, so the
 * list does not blank out every time a filter moves.
 */
export function useApi(url, { keepPrevious = true } = {}) {
  const [state, setState] = useState({ data: null, error: "", loading: Boolean(url) });
  const latest = useRef(0);

  const run = useCallback((targetUrl, { bypassCache = false } = {}) => {
    if (!targetUrl) {
      setState({ data: null, error: "", loading: false });
      return () => {};
    }

    const cached = bypassCache ? undefined : readCache(targetUrl);
    if (cached !== undefined) {
      setState({ data: cached, error: "", loading: false });
      return () => {};
    }

    const ticket = ++latest.current;
    const controller = new AbortController();
    setState((prev) => ({
      data: keepPrevious ? prev.data : null,
      error: "",
      loading: true
    }));

    apiGet(targetUrl, { signal: controller.signal })
      .then((data) => {
        if (ticket !== latest.current) return;
        writeCache(targetUrl, data);
        setState({ data, error: "", loading: false });
      })
      .catch((error) => {
        if (error.name === "AbortError" || ticket !== latest.current) return;
        setState({ data: null, error: error.message, loading: false });
      });

    return () => controller.abort();
  }, [keepPrevious]);

  useEffect(() => run(url), [url, run]);

  const reload = useCallback(() => {
    cache.delete(url);
    return run(url, { bypassCache: true });
  }, [url, run]);

  return { ...state, reload };
}

/** Debounce a fast-changing value (search box) before it reaches the network. */
export function useDebounced(value, ms = 260) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return debounced;
}
