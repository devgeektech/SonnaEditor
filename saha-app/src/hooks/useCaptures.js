// useCaptures — wraps GET /api/captures. The endpoint is read-only and
// idempotent; this hook just memoises the latest response and exposes a
// refetch for after a fine-tune run completes.

import { useCallback, useEffect, useState } from 'react';
import { fetchCaptures } from '../api/client.js';

export function useCaptures() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetchCaptures();
      setData(resp);
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refetch(); }, [refetch]);

  return { data, loading, error, refetch };
}
