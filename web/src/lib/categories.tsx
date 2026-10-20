// Icons and colours of the categories of elements (the second part of their identifier, lumos.<category>.Name).

import {
  Aperture,
  Box,
  Cable,
  Camera,
  CircleDashed,
  CircleDot,
  Cpu,
  FlaskConical,
  Image,
  Sigma,
  Split,
  Sun,
  Telescope,
  Wind,
  Workflow,
  type LucideIcon,
} from "lucide-react";

type Category = { icon: LucideIcon; color: string };

const CATEGORIES: Record<string, Category> = {
  sources: { icon: Sun, color: "#f59e0b" },
  scenes: { icon: Image, color: "#f97316" },
  telescopes: { icon: Telescope, color: "#6366f1" },
  apertures: { icon: CircleDashed, color: "#8b5cf6" },
  atmosphere: { icon: Wind, color: "#06b6d4" },
  optics: { icon: Aperture, color: "#3b82f6" },
  coronagraphs: { icon: CircleDot, color: "#a855f7" },
  fibers: { icon: Cable, color: "#14b8a6" },
  photonics: { icon: Cpu, color: "#10b981" },
  buses: { icon: Split, color: "#0ea5e9" },
  detectors: { icon: Camera, color: "#ef4444" },
  processing: { icon: Workflow, color: "#22c55e" },
  lab: { icon: FlaskConical, color: "#ec4899" },
  symbolic: { icon: Sigma, color: "#d946ef" },
};

/** Order of the categories in the palette: along the light path, from sources to processing. */
export const CATEGORY_ORDER = Object.keys(CATEGORIES);

export function category(name: string): Category {
  return CATEGORIES[name] ?? { icon: Box, color: "#64748b" };
}

export function categoryOf(typeId: string): string {
  const parts = typeId.split(".");
  return parts.length >= 3 ? parts[1] : "other";
}
