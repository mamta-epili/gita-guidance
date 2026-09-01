import { Component, ChangeDetectionStrategy, inject, signal, computed, effect } from '@angular/core';
import { Api, label } from '../core/api';
import { NextChar } from './next-char';
import { AttentionView } from './attention-view';
import { CausalMatrix } from './causal-matrix';
import { GenerateStream } from './generate-stream';

@Component({
  selector: 'lab-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [NextChar, AttentionView, CausalMatrix, GenerateStream],
  template: `
    <div class="wrap">
      <header>
        <h1><span class="om">ॐ</span>char-GPT inspector</h1>
        <p class="sub">
          A character-level transformer, written from scratch and trained on 124,135
          characters of the Bhagavad Gita. This page runs that model locally and shows what
          it is doing — the next-character distribution, and what each attention head is
          looking at.
        </p>

        @if (api.error(); as err) {
          <p class="warn" style="margin-top:14px">{{ err }}</p>
        }

        @if (api.info(); as i) {
          <div class="stats">
            @for (c of chips(); track c.k) {
              <span class="chip">{{ c.k }} <b>{{ c.v }}</b></span>
            }
          </div>
        }
      </header>

      <section>
        <h2>1 · One forward pass</h2>
        <p class="note">
          Type anything. Every keystroke runs a full forward pass and shows the model's
          prediction for the <em>next</em> character. This distribution is the model's
          entire output — generation is just sampling from it, appending, and running again.
        </p>
        <div class="panel">
          <textarea
            spellcheck="false"
            autocorrect="off"
            autocapitalize="off"
            data-gramm="false"
            data-gramm_editor="false"
            data-enable-grammarly="false"
            [value]="text()"
            (input)="text.set($any($event.target).value)"
          ></textarea>

          <div class="row">
            <label
              >temperature
              <input
                type="range" min="0.1" max="2" step="0.05"
                [value]="temp()"
                (input)="temp.set(+$any($event.target).value)"
              />
              <b style="color:var(--gold);font:12px var(--mono)">{{ temp().toFixed(2) }}</b>
            </label>
            @if (api.step()?.dropped_chars) {
              <span class="warn">
                {{ api.step()!.dropped_chars }} character(s) outside the model's vocabulary
                were dropped
              </span>
            }
          </div>

          <div class="grid2" style="margin-top:16px">
            <div>
              <h2 style="font-size:11px">Next character — top 12</h2>
              <next-char [step]="api.step()" />
            </div>
            <div>
              <h2 style="font-size:11px">Measurements</h2>
              @if (api.step(); as s) {
                <div class="metric">
                  <span>entropy</span>
                  <b
                    >{{ s.entropy_bits }} bits
                    <span style="color:var(--faint)">
                      of {{ s.max_entropy_bits }} ({{ entropyPct() }}%)</span
                    ></b
                  >
                </div>
                <div class="metric">
                  <span>context used</span>
                  <b>{{ s.context_used }} / {{ api.info()?.block_size }}</b>
                </div>
                <div class="metric">
                  <span>characters dropped by crop</span><b>{{ s.context_dropped }}</b>
                </div>
                <div class="metric"><span>forward pass</span><b>{{ s.ms }} ms</b></div>
              }
              <p class="note" style="margin-top:12px">
                Entropy is how undecided the model is, in bits. log₂(79) ≈ 6.30 means "no
                idea"; near 0 means near-certainty. Watch it collapse after a space, when
                only a few letters can plausibly start a word.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2>2 · What the attention heads are looking at</h2>
        <p class="note">
          Shading shows how much the <strong>final</strong> position paid to every earlier
          character when predicting the next one. Four layers, four heads each — sixteen
          views of the same text. Nothing ever attends to its right: that is the causal
          mask.
        </p>
        <div class="panel">
          <attention-view [step]="api.step()" [info]="api.info()" />
        </div>
      </section>

      <section>
        <h2>3 · Generate</h2>
        <p class="note">
          One character per forward pass, streamed at the speed it is produced. The text
          will have the shape of Arnold's blank verse and will mean nothing — the correct
          output for a 3.26M-parameter character model.
        </p>
        <div class="panel">
          <generate-stream [prompt]="text()" [temperature]="temp()" />
        </div>
      </section>

      <section>
        <h2>4 · Causality, as a picture</h2>
        <p class="note">
          Layer 0, head 0. Row <em>t</em> is what position <em>t</em> attended to. The upper
          triangle is black because <code>masked_fill(tril == 0, -inf)</code> made it exactly
          zero before the softmax.
        </p>
        <div class="panel"><causal-matrix [step]="api.step()" /></div>
      </section>

      <section>
        <h2>5 · The whole vocabulary</h2>
        <p class="note">
          Every symbol this model can represent — not a word among them. It learned "Arjuna"
          as seven statistical events. Anything outside this set is outside the model's
          universe, and gets silently dropped above.
        </p>
        <div class="panel">
          <div class="vocab">
            @for (c of api.info()?.vocab ?? []; track $index) {
              <span>{{ lbl(c) }}</span>
            }
          </div>
        </div>
      </section>

      <footer>
        <p>
          <strong>What this is.</strong> A demo of the foundations, running on your machine.
          No API calls, no network.
        </p>
        <p>
          <strong>What it is not.</strong> Not a question-answering system, and not usable as
          one. Its context window is 256 <em>characters</em> — one verse plus a short
          question fills it entirely, leaving no room to answer in.
        </p>
      </footer>
    </div>
  `,
  styles: [
    `
      header { border-bottom: 1px solid var(--line); padding-bottom: 18px; margin-bottom: 26px; }
      .stats { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
      .metric {
        display: flex; justify-content: space-between; font: 12px var(--mono);
        color: var(--dim); padding: 5px 0; border-bottom: 1px solid #221c17;
      }
      .metric b { color: var(--ink); font-weight: 500; }
      .vocab { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 6px; }
      .vocab span {
        font: 11px var(--mono); background: #0c0a09; border: 1px solid var(--line);
        border-radius: 2px; padding: 3px 5px; color: var(--dim);
      }
      footer {
        margin-top: 46px; padding-top: 18px; border-top: 1px solid var(--line);
        color: var(--faint); font-size: 13px;
      }
      code { font: 12px var(--mono); color: var(--gold); }
    `,
  ],
})
export class LabPage {
  readonly api = inject(Api);

  readonly text = signal('Arjuna said: my mind is');
  readonly temp = signal(1);

  lbl = label;

  readonly chips = computed(() => {
    const i = this.api.info();
    if (!i) return [];
    return [
      { k: 'params', v: i.params_m + 'M' },
      { k: 'vocab', v: i.vocab_size },
      { k: 'layers', v: i.n_layer },
      { k: 'heads', v: i.n_head },
      { k: 'n_embd', v: i.n_embd },
      { k: 'head_size', v: i.head_size },
      { k: 'context', v: i.block_size + ' chars' },
      { k: 'device', v: i.device },
    ];
  });

  readonly entropyPct = computed(() => {
    const s = this.api.step();
    if (!s) return '0';
    return ((s.entropy_bits / s.max_entropy_bits) * 100).toFixed(0);
  });

  private debounce: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    this.api.loadInfo();

    // Re-run the forward pass when text or temperature changes, debounced so a
    // fast typist doesn't queue one request per keystroke.
    effect(() => {
      const t = this.text();
      const temp = this.temp();
      if (this.debounce) clearTimeout(this.debounce);
      this.debounce = setTimeout(() => this.api.runStep(t, temp, true), 180);
    });
  }
}
