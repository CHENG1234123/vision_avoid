# Project guidance

## Obstacle avoidance report PowerPoint

When asked to create, regenerate, or resume the 单目视觉避障汇报 PowerPoint, run the existing generator first from the project root:

    node docs/gen_deck.cjs

This writes the editable nine-slide deck to `docs/单目视觉避障汇报.pptx`. If the dependency is missing, run `npm ci --prefix docs` and rerun the generator. If content changes are requested, update `docs/gen_deck.cjs` using verified project results and regenerate.

Confirm the target deck exists before doing further layout research. A probe deck is not the deliverable. Investigate only the specific error produced by the generator or a failed output check, then rerun once after correcting it. The generator and this guidance apply only to the report and do not change the ROS runtime.
