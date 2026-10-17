// Live updates from the server (WebSocket /api/ws): progress of the runs and journal entries.

import { useStore } from "./store";

let socket: WebSocket | null = null;
let retry = 500;
let stopped = false;

export function connectEvents(): void {
  stopped = false;
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${window.location.host}/api/ws`);
  socket.onopen = () => {
    retry = 500;
  };
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    const store = useStore.getState();
    if (message.type === "job") store.onJob(message.job);
    else if (message.type === "journal") store.onJournal(message.context, message.entry);
  };
  socket.onclose = () => {
    socket = null;
    if (stopped) return;
    window.setTimeout(connectEvents, retry);
    retry = Math.min(retry * 2, 10_000);
  };
}

export function disconnectEvents(): void {
  stopped = true;
  socket?.close();
}
