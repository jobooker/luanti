# Claude working agreement (luanti engine fork)

- **Plans come from `~/code/luanti-docs/spec/roadmap.md`.** Read it
  before starting work: it names the current step and the measurement
  that gates it. This repo is typically driven by Opus implementation
  sessions; planning and roadmap changes happen in Fable sessions in
  `~/code/luanti-docs`.
- **Required reading before touching the renderer:**
  `~/code/luanti-docs/spec/physics-contract.md` (how a change is
  judged) and `~/code/luanti-docs/spec/environment-laws.md` (platform
  and harness landmines — none will be re-derived by thinking).
- **Results flow back to the spec repo, not chat:** numbers and
  convictions into `spec/measured.md`, order changes into
  `spec/roadmap.md`. A step isn't done until its gate measurement is
  recorded there.

## Delegation rules (agreed 2026-08-23)

The lead session plans, reviews and records; implementation is farmed
out. Pick the tier by what a wrong answer costs, not by how big the diff
is:

- **Sonnet** — mechanical, fully-specified work with a verifier already
  in hand: renames, adding `exists()`/default registrations, running an
  existing script and pasting its output, formatting `measured.md`
  entries from numbers already taken, grep sweeps. If the brief cannot
  say how the result is checked, it is not a Sonnet task.
- **Opus** — anything that touches the renderer, the harness, or a
  referee; anything whose output is a *number that will be believed*.
  The brief must name the gate (which script, which arm, what
  red/green looks like) and the landmines from
  `spec/environment-laws.md` that apply.
- **Never delegated**: roadmap order, choosing what a referee asserts,
  deciding whether a measurement is retracted. Those stay with the lead.

Every brief carries: the branch, the files it may touch, the seat rule
(`bin/luanti --go` logs in as `claude` and KICKS whoever holds it — check
`pgrep -f "bin/luanti"` first; use `util/claude_look.sh --stop` only for
a seat you started), and "report what you ran and what it printed, not
what you concluded". Agents do not commit; the lead reviews the diff and
commits with the measurement in the message.
