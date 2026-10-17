// Operation of equipment (REQ-WEB-9): devices, their state, the actions of the elements and the emergency stop.
// Actions on equipment are confirmed in a dialog naming the device; the server checks the rights again.

import { AlertOctagon, Cpu, Link2, Play, RefreshCw, Search, ShieldAlert } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api, type Device, type DeviceStatus } from "../api";
import { parseQuantity } from "../lib/units";
import { runAction, useStore } from "../store";
import { Button, Modal } from "./Dialogs";
import { inputClass } from "./Fields";

function parseArgs(text: string): unknown[] {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const quantity = parseQuantity(part);
      if (quantity && !quantity.unit) return quantity.value;
      return part;
    });
}

export function OperationPanel() {
  const { t } = useTranslation();
  const info = useStore((s) => s.info);
  const context = useStore((s) => s.context);
  const [urls, setUrls] = useState("tcp://127.0.0.1:5100");
  const [found, setFound] = useState<Device[]>([]);
  const [devices, setDevices] = useState<DeviceStatus[]>([]);
  const [args, setArgs] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const operator = info?.role === "operator";
  const enabled = Boolean(info?.operation);

  const refresh = useCallback(async () => {
    if (!context) return;
    try {
      setDevices(await api.devices(context.id));
    } catch {
      setDevices([]);
    }
  }, [context]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const discover = async () => {
    setBusy(true);
    try {
      const list = urls.split(/[\s,]+/).filter(Boolean);
      setFound(await api.discover(list));
    } catch (error) {
      useStore.getState().fail(error);
    } finally {
      setBusy(false);
    }
  };

  const associate = async () => {
    if (!context) return;
    try {
      await api.associate(context.id);
      await refresh();
    } catch (error) {
      useStore.getState().fail(error);
    }
  };

  const actionable = (context?.elements ?? []).filter((e) => e.actions.length > 0);

  return (
    <div data-testid="operation" className="grid h-full grid-cols-[minmax(18rem,24rem)_1fr] gap-4 overflow-y-auto p-3 text-sm">
      <section className="space-y-2">
        <h3 className="flex items-center gap-1.5 text-xs font-semibold tracking-wider text-slate-500 uppercase">
          <Cpu className="size-3.5" />
          {t("operation.equipment")}
        </h3>
        {!enabled ? (
          <p className="flex gap-2 rounded-lg bg-slate-50 p-2 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">
            <ShieldAlert className="size-4 shrink-0 text-slate-400" />
            {t("operation.disabled")}
          </p>
        ) : !operator ? (
          <p className="flex gap-2 rounded-lg bg-amber-50 p-2 text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
            <ShieldAlert className="size-4 shrink-0" />
            {t("operation.operatorOnly")}
          </p>
        ) : (
          <>
            <label className="block text-xs text-slate-500">
              {t("operation.addresses")}
              <textarea data-testid="device-urls" rows={2} className={`${inputClass} mt-1 font-mono text-xs`} value={urls} onChange={(e) => setUrls(e.target.value)} />
            </label>
            <div className="flex gap-2">
              <Button testId="discover" onClick={() => void discover()} disabled={busy}>
                <Search className="size-4" />
                {t("operation.discover")}
              </Button>
              <Button testId="associate" onClick={() => void associate()} disabled={!found.length}>
                <Link2 className="size-4" />
                {t("operation.associate")}
              </Button>
            </div>
            <ul data-testid="discovered" className="space-y-0.5 font-mono text-xs">
              {found.map((device) => (
                <li key={device.url}>
                  {device.model} · {device.serial} · <span className="text-slate-400">{device.url}</span>
                </li>
              ))}
            </ul>
          </>
        )}
        <div className="flex items-center justify-between pt-1">
          <h3 className="text-xs font-semibold tracking-wider text-slate-500 uppercase">{t("operation.devices")}</h3>
          <button type="button" onClick={() => void refresh()} className="rounded p-1 text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800" title={t("operation.refresh")}>
            <RefreshCw className="size-3.5" />
          </button>
        </div>
        <table data-testid="devices" className="w-full text-xs">
          <tbody>
            {devices.length === 0 && (
              <tr>
                <td className="text-slate-400">{t("operation.noDevice")}</td>
              </tr>
            )}
            {devices.map((device) => (
              <tr key={device.element} className="border-b border-slate-100 dark:border-slate-800">
                <td className="py-1 font-mono">{device.element}</td>
                <td className="text-slate-500">{String(device.model ?? device.driver ?? "")}</td>
                <td>
                  <span
                    className={`rounded px-1.5 py-0.5 ${device.state === "stopped" ? "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300" : device.state === "busy" ? "bg-amber-100 text-amber-700" : "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"}`}
                  >
                    {String(device.state ?? "?")}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="space-y-2">
        <h3 className="text-xs font-semibold tracking-wider text-slate-500 uppercase">{t("operation.actions")}</h3>
        {actionable.length === 0 && <p className="text-xs text-slate-400">{t("operation.noAction")}</p>}
        {actionable.map((element) => (
          <div key={element.id} className="rounded-lg border border-slate-200 p-2 dark:border-slate-700">
            <div className="mb-1.5 flex items-center gap-2">
              <span className="font-mono text-xs font-semibold">{element.id}</span>
              <input
                className={`${inputClass} max-w-48 py-0.5 font-mono text-xs`}
                placeholder={t("operation.arguments")}
                value={args[element.id] ?? ""}
                onChange={(e) => setArgs({ ...args, [element.id]: e.target.value })}
                data-testid={`args-${element.id}`}
              />
            </div>
            <div className="flex flex-wrap gap-1">
              {element.actions.map((action) => (
                <button
                  key={action}
                  type="button"
                  data-testid={`action-${element.id}-${action}`}
                  disabled={info?.role === "viewer"}
                  onClick={() => void runAction(element.id, action, parseArgs(args[element.id] ?? ""))}
                  className="inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-1 font-mono text-xs hover:bg-accent-100 hover:text-accent-700 disabled:opacity-40 dark:bg-slate-800 dark:hover:bg-accent-900/40"
                >
                  <Play className="size-3" />
                  {action}
                </button>
              ))}
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}

export function EmergencyStop() {
  const { t } = useTranslation();
  const context = useStore((s) => s.context);
  if (!context) return null;
  const stop = async () => {
    try {
      await api.stop(context.id);
      useStore.getState().toast({ kind: "info", title: "stopped" });
    } catch (error) {
      useStore.getState().fail(error);
    }
  };
  return (
    <button
      type="button"
      data-testid="emergency-stop"
      onClick={() => void stop()}
      title={t("operation.stopHint")}
      className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-red-600 px-2.5 text-sm font-bold text-white shadow-sm ring-2 ring-red-300 hover:bg-red-700 dark:ring-red-900"
    >
      <AlertOctagon className="size-4" />
      {t("operation.stop")}
    </button>
  );
}

export function ConfirmAction({ element, action, device, onConfirm }: { element: string; action: string; device: string; onConfirm: () => void }) {
  const { t } = useTranslation();
  const close = () => useStore.getState().setDialog(null);
  return (
    <Modal title={t("operation.confirmTitle")} onClose={close} testId="confirm-action">
      <div className="space-y-4 p-5">
        <p className="flex gap-3 text-sm">
          <ShieldAlert className="size-6 shrink-0 text-amber-500" />
          <span>{t("operation.confirmText", { action: `${element}.${action}`, device })}</span>
        </p>
        <div className="flex justify-end gap-2">
          <Button onClick={close}>{t("common.cancel")}</Button>
          <Button
            primary
            testId="dialog-confirm"
            onClick={() => {
              close();
              onConfirm();
            }}
          >
            {t("operation.confirm")}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
