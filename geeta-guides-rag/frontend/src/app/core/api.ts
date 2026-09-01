import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

/** Architecture of the loaded checkpoint, read from the weights themselves. */
export interface ModelInfo {
  params: number;
  params_m: number;
  vocab_size: number;
  n_embd: number;
  n_head: number;
  n_layer: number;
  head_size: number;
  block_size: number;
  device: string;
  vocab: string[];
  checkpoint: string;
  ln_vocab: number;
}

export interface TopChar {
  char: string;
  prob: number;
}

export interface StepResult {
  context_used: number;
  context_dropped: number;
  entropy_bits: number;
  max_entropy_bits: number;
  top: TopChar[];
  /** attention[layer][head][position] — what the LAST position attended to. */
  attention?: number[][][];
  /** Full lower-triangular matrix for layer 0 head 0, capped at 48x48. */
  attention_matrix_l0h0?: number[][];
  ms: number;
  cleaned: string;
  dropped_chars: number;
  context_chars: string[];
}

export interface Rendering {
  text: string;
  translator: string;
  licence: string;
  redistributable: boolean;
}

export type Speaker = 'krishna' | 'arjuna' | 'sanjaya' | 'dhritarashtra';

export interface VerseHit {
  id: string;
  chapter: number;
  verse: number;
  citation: string;
  speaker: Speaker;
  speaker_name: string;
  sanskrit: string | null;
  iast: string | null;
  english: Rendering | null;
  hindi: Rendering | null;
  score: number | null;
  matched_lang: string | null;
  /** The verse this one answers — renders the 6:34 → 6:35 exchange as a pair. */
  asks?: VerseHit;
}

/**
 * Set when the answer was hand-chosen rather than retrieved. Carries its own
 * section headings, because "You are not the first" reads very differently from
 * "the same difficulty, as Arjuna put it".
 */
export interface Curated {
  reason: string;
  lead_heading_sa: string;
  lead_heading_en: string;
  follow_heading_sa: string;
  follow_heading_en: string;
  follow_note: string;
}

export interface GuidanceResult {
  curated: Curated | null;
  question: string;
  model?: string;
  /** Krishna's verses — the answer. */
  teaching: VerseHit[];
  /** Arjuna's / Sanjaya's verses — the question restated, shown separately. */
  dialogue: VerseHit[];
  verses: VerseHit[];
  pool_size?: number;
  top_score: number | null;
  score_note?: string;
  note?: string;
  ms?: number;
}

export interface GuidanceReady {
  ready: boolean;
  reason?: string;
  verses?: number;
  model?: string;
  dim?: number;
  rows?: number;
}

@Injectable({ providedIn: 'root' })
export class Api {
  private http = inject(HttpClient);

  readonly info = signal<ModelInfo | null>(null);
  readonly step = signal<StepResult | null>(null);
  readonly error = signal<string | null>(null);
  readonly busy = signal(false);

  async loadInfo(): Promise<void> {
    try {
      this.info.set(await firstValueFrom(this.http.get<ModelInfo>('/api/info')));
      this.error.set(null);
    } catch (e: unknown) {
      this.error.set(
        'Could not reach the backend. Is it running? `make backend` in another terminal.',
      );
    }
  }

  async runStep(text: string, temperature: number, attention = true): Promise<void> {
    this.busy.set(true);
    try {
      this.step.set(
        await firstValueFrom(
          this.http.post<StepResult>('/api/step', { text, temperature, attention }),
        ),
      );
      this.error.set(null);
    } catch {
      this.error.set('Forward pass failed.');
    } finally {
      this.busy.set(false);
    }
  }

  // -- guidance --------------------------------------------------------
  readonly guidance = signal<GuidanceResult | null>(null);
  readonly ready = signal<GuidanceReady | null>(null);
  readonly asking = signal(false);

  async checkReady(): Promise<void> {
    try {
      this.ready.set(await firstValueFrom(this.http.get<GuidanceReady>('/api/guidance/ready')));
    } catch {
      this.ready.set({ ready: false, reason: 'Backend not reachable.' });
    }
  }

  async ask(question: string, k = 5): Promise<void> {
    this.asking.set(true);
    try {
      this.guidance.set(
        await firstValueFrom(
          this.http.post<GuidanceResult>('/api/guidance', { question, k }),
        ),
      );
      this.error.set(null);
    } catch (e: any) {
      // 503 means the index isn't built; the detail carries the fix.
      this.error.set(e?.error?.detail ?? 'Retrieval failed.');
      this.guidance.set(null);
    } finally {
      this.asking.set(false);
    }
  }

  /**
   * Stream generated characters over SSE.
   *
   * EventSource callbacks run outside Angular's zone, which would normally mean
   * no change detection. Signals don't care — they notify their consumers
   * directly — so `onChar` can set a signal and the view updates. This is the
   * one place the signals-over-zone.js difference is load-bearing.
   */
  stream(
    prompt: string,
    n: number,
    temperature: number,
    onChar: (ch: string) => void,
    onDone: () => void,
  ): () => void {
    const url =
      `/api/stream?prompt=${encodeURIComponent(prompt)}` +
      `&n=${n}&temperature=${temperature}`;
    const es = new EventSource(url);

    es.onmessage = (ev) => {
      const d = JSON.parse(ev.data) as { type: string; char?: string };
      if (d.type === 'char' && d.char !== undefined) onChar(d.char);
      else if (d.type === 'done') {
        es.close();
        onDone();
      }
    };
    es.onerror = () => {
      es.close();
      onDone();
    };
    return () => {
      es.close();
      onDone();
    };
  }
}

/**
 * Attention weight -> CSS colour.
 *
 * The gamma of 0.55 is deliberate: raw attention weights on a long context are
 * mostly tiny, so a linear opacity ramp renders as a uniformly blank strip with
 * one bright square. Compressing the low end is what makes the pattern legible.
 */
export function heat(weight: number, max: number): string {
  if (!max) return 'transparent';
  const t = Math.pow(weight / max, 0.55);
  return `rgba(224,128,58,${(t * 0.92).toFixed(3)})`;
}

/** Printable label for a vocabulary character. */
export function label(c: string): string {
  if (c === '\n') return '⏎';
  if (c === ' ') return '␣';
  return c;
}
