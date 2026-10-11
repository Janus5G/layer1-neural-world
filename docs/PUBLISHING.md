# Publishing this experiment on GitHub

This ZIP **already contains the English source tree**. Do not add `node_modules`, `dist`, `.env`, or personal machine files.

1. Extract the ZIP into a new folder; keep the included directory structure.
2. Install Node.js 20+ and open a terminal in the extracted project folder.
3. Run `npm ci`, `npm run typecheck`, `npm test`, and `npm start`.
4. Verify the app at http://localhost:8787, the Pause/Reset controls, agent selection and English labels. Optional Ollama requires a separately installed local model.
5. On GitHub create an **empty** public repository, e.g. `layer1-neural-world` under your own account. Do not initialize it with a second README or license.
6. From the project folder, use these commands after setting Git author details:

```bash
git init
git add .
git commit -m "Publish Layer 1+ English artificial-life experiment"
git branch -M main
git remote add origin https://github.com/Janus5G/layer1-neural-world.git
git push -u origin main
```

The URL is only a **suggested address**; change it if the new repository has another name. If Git prompts for credentials, complete GitHub's normal authentication flow.

Suggested GitHub description:

> Experimental local artificial-life simulation with neural agents, reinforcement-learning-inspired training, curiosity, a mirror-mark task and optional Ollama narration. MIT licensed.

Suggested topics: `artificial-life`, `reinforcement-learning`, `neural-networks`, `ollama`, `typescript`, `agent-based-modeling`, `simulation`, `embodied-ai`.

The public release should state that **only Layer 1/1+ is implemented**. The later roadmap layers are available for independent contributors; the original author does not plan to continue development.
