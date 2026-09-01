# Ponytail — Codex CLI / Antigravity drop-in

Extracted from `ponytail-main.zip`. Six skills, already in the folder+SKILL.md
format both agents auto-discover — no plugin install, no marketplace, works
on free tier.

## Install (one step)

Copy `.agents/` and `AGENTS.md` into your project root. Done.

```
your-project/
├── AGENTS.md              ← always-on ruleset, loaded every turn
└── .agents/
    └── skills/
        ├── ponytail/SKILL.md          lazy senior dev mode (default)
        ├── ponytail-review/SKILL.md   over-engineering review of a diff
        ├── ponytail-audit/SKILL.md    whole-repo over-engineering audit
        ├── ponytail-debt/SKILL.md     harvest `ponytail:` shortcut comments
        ├── ponytail-gain/SKILL.md     measured-impact scoreboard
        └── ponytail-help/SKILL.md     quick reference card
```

- **Codex CLI** scans `.agents/skills` from cwd up to repo root automatically.
- **Antigravity** reads project-scope skills from `<project-root>/.agents/skills/`.
- Both also read root `AGENTS.md` as always-on context — this alone gives you
  the core lazy-dev ruleset even before any skill fires.

Global instead of per-project:
- Codex CLI: `~/.agents/skills/` (or `~/.codex/skills/`)
- Antigravity: `~/.gemini/config/skills/`

## Use it

Skills load only when the task matches their description (progressive
disclosure — no context cost until triggered).

- **Implicit**: just describe the task normally ("add a cache for these API
  responses") — `ponytail` fires on its own.
- **Explicit**:
  - Codex: type `$ponytail`, `$ponytail-review`, `$ponytail-audit`, `$ponytail-debt`, `$ponytail-gain`, `$ponytail-help` in the prompt.
  - Antigravity: type `/ponytail`, `/ponytail-review`, etc. in the agent chat (Antigravity surfaces skills as slash-style commands).

Levels (persist for the session): `/ponytail lite` (build it, name the lazier
alternative), `/ponytail` (default — YAGNI → stdlib → native → one line →
minimum), `/ponytail ultra` (challenge the requirement itself). Turn off:
"stop ponytail" / "normal mode".

## What each one does

| Skill | Trigger it for |
|---|---|
| `ponytail` | Every coding task — writes/reviews/refactors the smallest correct diff. |
| `ponytail-review` | "review this diff for over-engineering" |
| `ponytail-audit` | "audit this repo for bloat" |
| `ponytail-debt` | "what shortcuts did we defer" (greps `ponytail:` comments) |
| `ponytail-gain` | "show ponytail's measured impact" (benchmark scoreboard) |
| `ponytail-help` | "ponytail help" (this table, plus config) |

Source: github.com/DietrichGebert/ponytail
