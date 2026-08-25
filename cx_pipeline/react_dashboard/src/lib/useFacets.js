import { useEffect, useMemo, useRef, useState } from "react";
import { apiGetCached } from "./api.js";
import { ALL_FACETS } from "./facets.js";
import { filterQuery } from "./urlState.js";

/* The API counts facets under *all* active filters, including a facet's own
   dimension. That makes OR-within-a-dimension unreachable: pick "bad" and every
   other Experience option drops to zero and disappears, so you can never widen
   the selection to "bad OR neutral".
 *
 * Standard faceted search excludes a dimension from its own counts. We
 * reproduce that here by re-querying once per *active* dimension with that
 * dimension dropped, and overriding just those counts. Users typically have one
 * to three dimensions active, the responses are small, and api.js memoises them,
 * so this costs a handful of cached requests rather than a redesign.
 *
 * If the facets endpoint later grows per-dimension exclusion server-side, delete
 * the extra passes below and keep the base call. */

export function useFacets(state, { enabled = true } = {}) {
  const [facets, setFacets] = useState(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState("");
  const latest = useRef(0);

  const activeDimensions = useMemo(
    () => ALL_FACETS.filter((facet) => (state[facet.key] || []).length > 0),
    [state]
  );

  const baseQuery = useMemo(() => filterQuery(state), [state]);
  const exclusionQueries = useMemo(
    () => activeDimensions.map((facet) => ({
      facet: facet.facet,
      query: filterQuery({ ...state, [facet.key]: [] })
    })),
    [activeDimensions, state]
  );

  const signature = `${baseQuery}||${exclusionQueries.map((item) => item.facet).join(",")}`;

  useEffect(() => {
    if (!enabled) {
      setFacets(null);
      setLoading(false);
      return undefined;
    }

    const ticket = ++latest.current;
    const controller = new AbortController();
    setLoading(true);
    setError("");

    const requests = [
      apiGetCached(`/api/tickets/facets?${baseQuery}`, { signal: controller.signal }),
      ...exclusionQueries.map((item) =>
        apiGetCached(`/api/tickets/facets?${item.query}`, { signal: controller.signal })
          .then((data) => ({ facet: item.facet, data }))
      )
    ];

    Promise.all(requests)
      .then(([base, ...widened]) => {
        if (ticket !== latest.current) return;
        const merged = { ...(base.facets || {}) };
        for (const { facet, data } of widened) {
          if (data?.facets?.[facet]) merged[facet] = data.facets[facet];
        }
        setFacets(merged);
        setLoading(false);
      })
      .catch((cause) => {
        if (cause.name === "AbortError" || ticket !== latest.current) return;
        setError(cause.message);
        setLoading(false);
      });

    return () => controller.abort();
  }, [signature, enabled]); // eslint-disable-line react-hooks/exhaustive-deps

  return { facets, loading, error };
}
