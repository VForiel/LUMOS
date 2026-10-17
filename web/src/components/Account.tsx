// Accounts (REQ-WEB-9): log in with a password or by signing a challenge with an SSH key; show the role.

import { Check, Copy, KeyRound, LogIn, LogOut, UserRound } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { api, ApiError } from "../api";
import { useStore } from "../store";
import { Button, Modal } from "./Dialogs";
import { inputClass } from "./Fields";

async function reloadInfo(): Promise<void> {
  const store = useStore.getState();
  useStore.setState({ info: await api.info() });
  await store.refresh();
}

export function AccountMenu() {
  const { t } = useTranslation();
  const info = useStore((s) => s.info);
  const [open, setOpen] = useState(false);
  if (!info?.auth.enabled) return null;
  if (!info.user) {
    return (
      <button
        type="button"
        data-testid="login"
        onClick={() => useStore.getState().setDialog({ kind: "login" })}
        className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-slate-200 px-2.5 text-sm font-medium hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800"
      >
        <LogIn className="size-4" />
        {t("account.login")}
      </button>
    );
  }
  return (
    <div className="relative">
      <button
        type="button"
        data-testid="user"
        onClick={() => setOpen(!open)}
        className="inline-flex h-8 items-center gap-1.5 rounded-lg px-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-800"
      >
        <UserRound className="size-4" />
        <span className="font-medium">{info.user}</span>
        <span className="rounded bg-slate-100 px-1.5 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">{t(`account.roles.${info.role}`)}</span>
      </button>
      {open && (
        <div className="absolute right-0 z-30 mt-1 w-44 rounded-lg border border-slate-200 bg-white p-1 shadow-lg dark:border-slate-700 dark:bg-slate-900">
          <button
            type="button"
            data-testid="logout"
            onClick={async () => {
              setOpen(false);
              await api.logout().catch(() => undefined);
              window.location.reload();
            }}
            className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            <LogOut className="size-4" />
            {t("account.logout")}
          </button>
        </div>
      )}
    </div>
  );
}

export function LoginDialog() {
  const { t } = useTranslation();
  const methods = useStore((s) => s.info?.auth.methods ?? []);
  const [tab, setTab] = useState<"password" | "ssh">("password");
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [challenge, setChallenge] = useState<string | null>(null);
  const [signature, setSignature] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const close = () => useStore.getState().setDialog(null);
  const fail = (e: unknown) => setError(e instanceof ApiError ? `${e.message}${e.hint ? ` — ${e.hint}` : ""}` : String(e));
  const done = async () => {
    close();
    await reloadInfo();
  };
  const command = challenge ? `echo -n "${challenge}" | ssh-keygen -Y sign -n lumos -f ~/.ssh/id_ed25519` : "";
  const tabClass = (active: boolean) =>
    `flex-1 rounded-md px-3 py-1.5 text-sm font-medium ${active ? "bg-white shadow-sm dark:bg-slate-700" : "text-slate-500"}`;

  return (
    <Modal title={t("account.login")} onClose={close} testId="login-dialog">
      <form
        className="space-y-3 p-5"
        onSubmit={async (event) => {
          event.preventDefault();
          setError(null);
          try {
            if (tab === "password") await api.login(user, password);
            else if (challenge) await api.sshLogin(user, challenge, signature);
            await done();
          } catch (e) {
            fail(e);
            if (tab === "ssh") setChallenge(null);
          }
        }}
      >
        {methods.includes("ssh") && (
          <div className="flex gap-1 rounded-lg bg-slate-100 p-1 dark:bg-slate-800">
            <button type="button" className={tabClass(tab === "password")} onClick={() => setTab("password")} data-testid="login-tab-password">
              {t("account.password")}
            </button>
            <button type="button" className={tabClass(tab === "ssh")} onClick={() => setTab("ssh")} data-testid="login-tab-ssh">
              <KeyRound className="mr-1 inline size-4" />
              {t("account.sshKey")}
            </button>
          </div>
        )}
        <label className="block text-sm">
          <span className="mb-1 block text-slate-600 dark:text-slate-300">{t("account.user")}</span>
          <input data-testid="login-user" className={inputClass} value={user} autoComplete="username" onChange={(e) => setUser(e.target.value)} />
        </label>
        {tab === "password" ? (
          <label className="block text-sm">
            <span className="mb-1 block text-slate-600 dark:text-slate-300">{t("account.password")}</span>
            <input data-testid="login-password" type="password" autoComplete="current-password" className={inputClass} value={password} onChange={(e) => setPassword(e.target.value)} />
          </label>
        ) : !challenge ? (
          <Button
            testId="login-challenge"
            onClick={async () => {
              setError(null);
              try {
                setChallenge((await api.challenge(user)).challenge);
              } catch (e) {
                fail(e);
              }
            }}
            disabled={!user}
          >
            {t("account.getChallenge")}
          </Button>
        ) : (
          <div className="space-y-2 text-sm">
            <p className="text-slate-600 dark:text-slate-300">{t("account.signHint")}</p>
            <div className="relative">
              <pre data-testid="ssh-command" className="overflow-x-auto rounded-lg bg-slate-950 p-2 pr-9 font-mono text-xs text-slate-100">
                {command}
              </pre>
              <button
                type="button"
                className="absolute top-1.5 right-1.5 rounded p-1 text-slate-400 hover:text-white"
                onClick={() => void navigator.clipboard?.writeText(command).then(() => setCopied(true))}
                aria-label={t("dialog.python.copy")}
              >
                {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
              </button>
            </div>
            <p className="text-xs text-slate-500">{t("account.loginHint")}</p>
            <textarea data-testid="login-signature" rows={5} className={`${inputClass} font-mono text-xs`} placeholder="-----BEGIN SSH SIGNATURE-----" value={signature} onChange={(e) => setSignature(e.target.value)} />
          </div>
        )}
        {error && <p role="alert" className="text-sm text-red-600 dark:text-red-400">{error}</p>}
        <div className="flex justify-end gap-2">
          <Button onClick={close}>{t("common.cancel")}</Button>
          <Button primary type="submit" testId="login-submit" disabled={!user || (tab === "ssh" && (!challenge || !signature))}>
            {t("account.login")}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
