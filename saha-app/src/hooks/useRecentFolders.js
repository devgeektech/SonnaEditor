// useRecentFolders — GET /api/folders/recent. Refetched after a process job
// starts so the UI reflects the just-recorded entry without a full reload.

import { useCallback, useEffect, useState } from 'react';
import { listRecentFolders } from '../api/client.js';

export function useRecentFolders() {
  const [folders, setFolders] = useState([]);
  const [loading, setLoading] = useState(true);

  const refetch = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listRecentFolders();
      setFolders(Array.isArray(data) ? data : []);
    } catch {
      // recent folders are best-effort; an empty list is a fine fallback
      setFolders([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refetch(); }, [refetch]);

  return { folders, loading, refetch };
}
