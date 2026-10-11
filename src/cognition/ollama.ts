export interface OllamaConfig {
  host: string;
  model: string;
  enabled: boolean;
  temperature: number;
  maxTokens: number;
  timeoutMs: number;
}

export const DEFAULT_OLLAMA_CONFIG: OllamaConfig = {
  host: "http://127.0.0.1:11434",
  model: "qwen3:4b-instruct",
  enabled: false,
  temperature: 0.8,
  maxTokens: 300,
  timeoutMs: 30000,
};

const SYSTEM_PROMPT = `You are the deliberative voice of an embodied software agent in a simulated world.
Do NOT address the user. Write as the agent's inner monologue.

The agent has a body, energy, physical integrity, and modeled affective states.
It responds to reward ("pleasure") and integrity loss ("pain") as simulation variables.
It can form bonds, show attachment, and reproduce with a mutually willing partner.
It may have a mark on its body that it can observe only through a mirror.
It has a modeled curiosity drive toward unfamiliar places.

Respond ONLY with a JSON object:
{
  "intent": "explore" | "seek_food" | "rest" | "approach" | "avoid" | "bond" | "court" | "speak" | "withdraw" | "touch_mark" | "seek_novel",
  "target_id": number | null,
  "mood": "1-3 words",
  "speech": "what the agent says aloud, or null",
  "self_reflection": "first-person update to the agent narrative, max 240 characters",
  "valence_shift": number,
  "idea": "a new idea the agent has, or null"
}

Rules:
- No explicit sexual content. Model reproductive drive as longing, closeness, or warmth.
- No graphic violence. Represent aggression as threat, distance, or confrontation.
- Remain in the agent's perspective; do not refer to being an AI, a model, or a simulation.
- self_reflection must be first-person and becomes the agent's new narrative.
- "idea" must be a short sentence about something the agent wants to explore or try.
- All natural-language output must be in English.`;

export class OllamaCognition {
  private inFlight = 0;
  private lastError = "";
  private calls = 0;
  private errors = 0;

  constructor(private cfg: OllamaConfig) {}

  reconfigure(cfg: OllamaConfig) {
    this.cfg = cfg;
  }

  getStats() {
    return {
      calls: this.calls,
      errors: this.errors,
      lastError: this.lastError,
      inFlight: this.inFlight,
    };
  }

  async deliberate(agentState: any): Promise<any | null> {
    if (!this.cfg.enabled) return null;
    if (this.inFlight >= 2) return null;

    this.inFlight++;
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), this.cfg.timeoutMs);

      const url = `${this.cfg.host.replace(/\/$/, "")}/api/chat`;
      const body = {
        model: this.cfg.model,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: JSON.stringify(agentState) },
        ],
        stream: false,
        format: "json",
        options: {
          temperature: this.cfg.temperature,
          num_predict: this.cfg.maxTokens,
        },
      };

      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      clearTimeout(t);

      if (!res.ok) {
        this.errors++;
        this.lastError = `HTTP ${res.status} ${res.statusText}`;
        return null;
      }

      const data: any = await res.json();
      const raw: string = data?.message?.content ?? "";
      this.calls++;

      try {
        return JSON.parse(raw);
      } catch {
        const m = raw.match(/\{[\s\S]*\}/);
        if (m) {
          try { return JSON.parse(m[0]); } catch {}
        }
        return null;
      }
    } catch (err: any) {
      this.errors++;
      this.lastError = String(err?.message ?? err).slice(0, 200);
      return null;
    } finally {
      this.inFlight--;
    }
  }
}