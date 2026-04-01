# Hanabi

Browser-based Hanabi card game (React + Vite + TypeScript).

## Architecture

- `src/engine/` — Pure game logic, no React imports. Designed to be lifted onto a server for multiplayer later.
- `src/components/` — React UI components with CSS Modules.
- `src/hooks/useGame.ts` — Bridges engine to React via useReducer, schedules bot turns.

## Development

```
npm install
npm run dev
```

## Conventions

- Always git commit after completing a milestone or meaningful chunk of work.
- Engine code must stay framework-agnostic (no React imports in `src/engine/`).
- Use `import type` for type-only imports (verbatimModuleSyntax is enabled).
- Dark theme UI with CSS Modules for styling.
