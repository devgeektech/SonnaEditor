// useProfiles — GET /api/profiles + activate. Returns the active profile
// derived from is_active rather than tracked separately, so refetch after
// activate keeps the source of truth on the server.

import { useCallback, useEffect, useState } from 'react';
import { activateProfile, listProfiles } from '../api/client.js';

export function useProfiles() {
  const [profiles, setProfiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listProfiles();
      setProfiles(Array.isArray(data) ? data : []);
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refetch(); }, [refetch]);

  const activate = useCallback(async (profileId) => {
    await activateProfile(profileId);
    await refetch();
  }, [refetch]);

  const activeProfile = profiles.find((p) => p.is_active) || null;

  return { profiles, activeProfile, loading, error, refetch, activate };
}
