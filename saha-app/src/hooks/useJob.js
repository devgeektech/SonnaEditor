// useJob — owns one process job's full lifecycle.
//
// Source of truth: the snapshot field. It's seeded from POST /api/process and
// then updated either by websocket messages (live) or by polling (fallback
// when the websocket has exhausted its 3 reconnect attempts). The hook never
// blends the two simultaneously — wsStatus drives which feed is authoritative.
//
// Public API:
//   const job = useJob({ onError })
//   job.current   → null | { id, snapshot, wsStatus }
//   job.start(req) → POST /api/process, opens stream, returns the new snapshot
//   job.cancel()   → POST /api/jobs/{id}/cancel
//   job.reset()    → clear current (e.g. "Process another folder")

import { useCallback, useEffect, useRef, useState } from 'react';
import { cancelJob, getJob, jobStreamUrl, startProcess } from '../api/client.js';
import { connectJobStream } from '../api/websocket.js';

const POLL_INTERVAL_MS = 1000;

const TERMINAL_STATES = new Set(['complete', 'cancelled', 'failed']);
const isTerminal = (state) => TERMINAL_STATES.has(state);

export function useJob({ onError } = {}) {
  const [current, setCurrent] = useState(null);

  // Refs for the lifecycle handles — these don't trigger re-renders.
  const streamRef = useRef(null);
  const pollTimerRef = useRef(null);
  const currentIdRef = useRef(null);

  // Photo / epoch progress messages are coalesced via requestAnimationFrame so
  // a 1000-photo shoot that fires 1000 messages in <1s flushes at ~60Hz instead
  // of overwhelming React's scheduler. Terminal messages flush synchronously
  // so the final state isn't lost on tab-blur or rAF skip.
  const pendingSnapshotRef = useRef(null);
  const pendingEpochRef = useRef(null);
  const rafIdRef = useRef(null);

  const stopFeeds = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    if (rafIdRef.current !== null) {
      if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(rafIdRef.current);
      else clearTimeout(rafIdRef.current);
      rafIdRef.current = null;
    }
    pendingSnapshotRef.current = null;
    pendingEpochRef.current = null;
  }, []);

  // Apply a websocket message to the current snapshot. See pending* ref
  // comment above for the rAF coalescing rationale.

  const flushPending = useCallback(() => {
    rafIdRef.current = null;
    const snapPatch = pendingSnapshotRef.current;
    const epochPatch = pendingEpochRef.current;
    pendingSnapshotRef.current = null;
    pendingEpochRef.current = null;
    if (!snapPatch && !epochPatch) return;
    setCurrent((prev) => {
      if (!prev) return prev;
      const snapshot = (snapPatch || epochPatch)
        ? { ...prev.snapshot, ...(snapPatch || {}), ...(epochPatch || {}) }
        : prev.snapshot;
      return { ...prev, snapshot };
    });
  }, []);

  const scheduleFlush = useCallback(() => {
    if (rafIdRef.current !== null) return;
    if (typeof requestAnimationFrame === 'function') {
      rafIdRef.current = requestAnimationFrame(flushPending);
    } else {
      // setTimeout fallback for non-browser environments (test harness).
      rafIdRef.current = setTimeout(flushPending, 16);
    }
  }, [flushPending]);

  const applyMessage = useCallback((msg) => {
    if (msg.type === 'job_snapshot') {
      const { type, ...snap } = msg;
      setCurrent((prev) => prev
        ? { ...prev, snapshot: { ...prev.snapshot, ...snap } }
        : prev);
      return;
    }
    if (msg.type === 'photo_prepared') {
      pendingSnapshotRef.current = {
        state: 'running',
        photos_total: msg.photos_total,
        photos_prepared: msg.photos_prepared,
        photos_per_sec: msg.photos_per_sec,
        eta_seconds: msg.eta_seconds,
        current_photo: msg.name,
      };
      scheduleFlush();
      return;
    }
    if (msg.type === 'photo_complete') {
      pendingSnapshotRef.current = {
        state: 'running',
        photos_total: msg.photos_total,
        photos_processed: msg.photos_processed,
        photos_per_sec: msg.photos_per_sec,
        eta_seconds: msg.eta_seconds,
        current_photo: msg.name,
      };
      scheduleFlush();
      return;
    }
    if (msg.type === 'epoch_complete') {
      pendingEpochRef.current = {
        state: 'running',
        epochs_completed: (msg.epoch ?? 0) + 1,
        current_epoch: msg.epoch,
        train_loss: msg.train_loss,
        val_loss: msg.val_loss,
      };
      scheduleFlush();
      return;
    }
    if (msg.type === 'job_complete' || msg.type === 'job_cancelled' || msg.type === 'job_failed') {
      // Flush any pending progress first so the terminal snapshot doesn't
      // overwrite the final photo update.
      if (rafIdRef.current !== null) {
        if (typeof cancelAnimationFrame === 'function') {
          cancelAnimationFrame(rafIdRef.current);
        } else {
          clearTimeout(rafIdRef.current);
        }
        rafIdRef.current = null;
        flushPending();
      }
      const { type, ...snap } = msg;
      setCurrent((prev) => prev
        ? { ...prev, snapshot: { ...prev.snapshot, ...snap } }
        : prev);
    }
  }, [flushPending, scheduleFlush]);

  const startPolling = useCallback(() => {
    if (pollTimerRef.current) return;
    pollTimerRef.current = setInterval(async () => {
      const id = currentIdRef.current;
      if (!id) return;
      try {
        const snap = await getJob(id);
        setCurrent((prev) => prev && prev.id === id
          ? { ...prev, snapshot: snap }
          : prev);
        if (isTerminal(snap.state)) {
          stopFeeds();
        }
      } catch {
        // transient — keep polling
      }
    }, POLL_INTERVAL_MS);
  }, [stopFeeds]);

  // start accepts an optional `requestFn` so the profile view can drive
  // POST /api/finetune through the same lifecycle. Defaults to startProcess
  // so existing editor calls (which omit requestFn) keep working unchanged.
  const start = useCallback(async (req, { requestFn } = {}) => {
    stopFeeds();

    const requester = requestFn || startProcess;
    let ack;
    try {
      ack = await requester(req);
    } catch (e) {
      if (onError) onError({ source: 'start', message: e.message });
      throw e;
    }

    // Seed snapshot from the first GET so initial fields (folder_path,
    // photos_total, etc.) are populated immediately.
    let snap;
    try {
      snap = await getJob(ack.job_id);
    } catch {
      snap = { job_id: ack.job_id, kind: 'process', state: ack.state || 'queued',
               photos_processed: 0, photos_per_sec: 0, eta_seconds: 0 };
    }

    currentIdRef.current = ack.job_id;
    setCurrent({
      id: ack.job_id,
      snapshot: snap,
      wsStatus: 'connecting',
    });

    streamRef.current = connectJobStream({
      url: jobStreamUrl(ack.job_id),
      onMessage: applyMessage,
      onStatus: (status) => {
        setCurrent((prev) => prev && prev.id === ack.job_id
          ? { ...prev, wsStatus: status }
          : prev);
      },
      onFallback: () => {
        startPolling();
      },
    });

    return snap;
  }, [applyMessage, onError, startPolling, stopFeeds]);

  const cancel = useCallback(async () => {
    const id = currentIdRef.current;
    if (!id) return;
    try {
      const snap = await cancelJob(id);
      setCurrent((prev) => prev && prev.id === id
        ? { ...prev, snapshot: snap }
        : prev);
    } catch (e) {
      if (onError) onError({ source: 'cancel', message: e.message });
    }
  }, [onError]);

  const reset = useCallback(() => {
    stopFeeds();
    currentIdRef.current = null;
    setCurrent(null);
  }, [stopFeeds]);

  // Stop feeds when the snapshot transitions to terminal (websocket path).
  useEffect(() => {
    if (current && isTerminal(current.snapshot.state)) {
      // The websocket has already closed itself; clear refs so reset() is a no-op.
      if (streamRef.current) { streamRef.current.close(); streamRef.current = null; }
      if (pollTimerRef.current) { clearInterval(pollTimerRef.current); pollTimerRef.current = null; }
    }
  }, [current?.snapshot?.state]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup on unmount.
  useEffect(() => () => stopFeeds(), [stopFeeds]);

  return { current, start, cancel, reset };
}

export { isTerminal };
