// Step-by-step tutorial overlaid on the interface (REQ-WEB-7, driver.js): from a preset to its results.

import { driver, type DriveStep } from "driver.js";
import "driver.js/dist/driver.css";
import { api } from "./api";
import i18n from "./i18n";
import { useStore } from "./store";

const PRESET = "first_simulation";

function waitFor(check: () => boolean, timeout = 60_000): Promise<void> {
  return new Promise((resolve) => {
    const started = Date.now();
    const poll = () => {
      if (check() || Date.now() - started > timeout) resolve();
      else window.setTimeout(poll, 150);
    };
    poll();
  });
}

function step(key: string, element?: string, side: "left" | "right" | "top" | "bottom" = "bottom"): DriveStep {
  return {
    element,
    popover: {
      title: i18n.t(`tour.${key}.title`),
      description: i18n.t(`tour.${key}.text`),
      side,
      align: "center",
    },
  };
}

/** Load the tutorial preset (unless open) and walk through the interface. */
export async function startTour(): Promise<void> {
  const store = useStore.getState();
  store.setDialog(null);
  if (!store.context || !store.context.name.toLowerCase().includes("first")) {
    try {
      await store.open(await api.loadPreset(PRESET));
    } catch (error) {
      store.fail(error);
      return;
    }
  }
  await waitFor(() => document.querySelector(".react-flow__node") !== null, 10_000);
  const firstNode = () => document.querySelector<HTMLElement>(".react-flow__node");
  const canRun = ["simulator", "operator"].includes(useStore.getState().info?.role ?? "");

  const steps: DriveStep[] = [
    step("welcome"),
    step("palette", '[data-tour="palette"]', "right"),
    step("canvas", '[data-tour="canvas"]', "left"),
    {
      ...step("block", ".react-flow__node", "bottom"),
      onHighlightStarted: () => {
        const id = firstNode()?.dataset.id;
        if (id) useStore.getState().select([id]);
      },
    },
    step("inspector", '[data-tour="inspector"]', "left"),
  ];
  if (canRun) {
    steps.push({
      ...step("run", '[data-tour="run"]', "bottom"),
      popover: {
        ...step("run").popover,
        onNextClick: async (_element, _step, { driver: tour }) => {
          const button = document.querySelector<HTMLButtonElement>(".driver-popover-next-btn");
          if (button?.disabled) return;
          if (button) {
            button.disabled = true;
            button.textContent = i18n.t("results.computing");
          }
          const state = useStore.getState();
          state.select([]);
          await state.run();
          await waitFor(() => {
            const current = useStore.getState();
            const job = current.lastJob ? current.jobs[current.lastJob] : undefined;
            return Boolean(job && job.status !== "queued" && job.status !== "running");
          });
          useStore.getState().setBottom("results");
          await waitFor(() => document.querySelector('[data-testid="output-card"]') !== null, 10_000);
          tour.moveNext();
        },
      },
    });
    steps.push(step("results", '[data-tour="results"]', "top"));
    steps.push(step("inspect", ".react-flow__node", "bottom"));
  }
  steps.push(step("files", '[data-tour="files"]', "bottom"));
  steps.push(step("help", '[data-tour="help"]', "bottom"));
  steps.push(step("done"));

  const tour = driver({
    showProgress: true,
    allowClose: true,
    overlayOpacity: 0.55,
    stagePadding: 6,
    stageRadius: 12,
    nextBtnText: i18n.t("tour.next"),
    prevBtnText: i18n.t("tour.previous"),
    doneBtnText: i18n.t("tour.finish"),
    progressText: i18n.t("tour.progress").replace("__current__", "{{current}}").replace("__total__", "{{total}}"),
    steps,
  });
  tour.drive();
}
