import path from "node:path";
import fs from "node:fs/promises";
const SKILL_DIR = "/home/allen/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations";
const workspaceDir = "/home/allen/Documents/Material/AML/prject_final";
const candidatePath = path.join(workspaceDir, ".ppt_build", "candidate.pptx");
const finalPath = path.join(workspaceDir, "presentation_output", "duplicate_question_detection_presentation_v3.pptx");
const { finalizePresentation } = await import(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs"));
const result = await finalizePresentation({
  explicitTotalSlideCount: 5,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: "/home/allen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3",
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
  fontPolicy: { basis: "design", families: ["Aptos"] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(workspaceDir, ".ppt_build", "validation-v3.json"),
});
console.log(JSON.stringify(result, null, 2));
