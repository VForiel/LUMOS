// The interface is bilingual (step P14.4, REQ-WEB-8): every text exists in English and in French, with the
// same interpolated variables.
import { describe, expect, it } from "vitest";
import en from "../../web/src/locales/en.json";
import fr from "../../web/src/locales/fr.json";

type Tree = { [key: string]: string | Tree };

function flatten(tree: Tree, prefix = ""): Record<string, string> {
  const flat: Record<string, string> = {};
  for (const [key, value] of Object.entries(tree)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (typeof value === "string") flat[path] = value;
    else Object.assign(flat, flatten(value, path));
  }
  return flat;
}

const variables = (text: string) => [...text.matchAll(/\{\{\s*(\w+)\s*\}\}/g)].map((m) => m[1]).sort();

describe("translations", () => {
  const english = flatten(en as Tree);
  const french = flatten(fr as Tree);
  it("have the same keys", () => {
    expect(Object.keys(french).sort()).toEqual(Object.keys(english).sort());
  });
  it("have no empty text", () => {
    for (const [key, text] of Object.entries({ ...english, ...french })) expect(text.trim(), key).not.toBe("");
  });
  it("keep the variables", () => {
    for (const key of Object.keys(english)) expect(variables(french[key] ?? ""), key).toEqual(variables(english[key]));
  });
  it("are really translated", () => {
    const same = Object.keys(english).filter((key) => english[key] === french[key] && english[key].length > 12);
    expect(same).toEqual([]);
  });
});
