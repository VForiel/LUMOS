// Callback of the OpenRouter login (OAuth PKCE): kept apart from the assistant so that the interface does not
// load the clients of the language models before the assistant panel opens.

export const OPENROUTER_CALLBACK = "/oauth/openrouter";

/** In the popup of the OpenRouter login: give the code to the window that opened it, then close. */
export function finishOpenRouterLogin(): boolean {
  if (window.location.pathname !== OPENROUTER_CALLBACK) return false;
  const query = new URLSearchParams(window.location.search);
  window.opener?.postMessage({ type: "lumos-openrouter", code: query.get("code"), state: query.get("state") }, window.location.origin);
  window.close();
  return true;
}
