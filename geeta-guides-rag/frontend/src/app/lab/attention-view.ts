import { Component, ChangeDetectionStrategy, input, signal, computed } from '@angular/core';
import { StepResult, ModelInfo, heat } from '../core/api';

/**
 * What the final position attended to, per layer and per head.
 *
 * Only the LAST row of each attention matrix is shown, because that is the row
 * that actually produced the prediction the model just made. The full matrix
 * lives in <causal-matrix>.
 */
@Component({
  selector: 'attention-view',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="row" style="margin-top:0">
      <label
        >layer
        <select [value]="layer()" (change)="layer.set(+$any($event.target).value)">
          @for (i of layers(); track i) {
            <option [value]="i">{{ i }}</option>
          }
        </select>
      </label>
      <label
        >head
        <select
          [value]="head()"
          [disabled]="average()"
          (change)="head.set(+$any($event.target).value)"
        >
          @for (i of heads(); track i) {
            <option [value]="i">{{ i }}</option>
          }
        </select>
      </label>
      <label>
        <input type="checkbox" [checked]="average()" (change)="average.set(!average())" />
        average all heads in this layer
      </label>
    </div>

    <div class="attn">
      @for (cell of cells(); track cell.i) {
        <span
          [class.cur]="cell.last"
          [style.background]="cell.bg"
          [title]="'pos ' + cell.i + ' · ' + (cell.w * 100).toFixed(2) + '%'"
          >{{ cell.txt }}</span
        >
      }
    </div>
  `,
  styles: [
    `
      .attn {
        font: 13px/2.1 var(--mono);
        word-break: break-all;
        margin-top: 10px;
      }
      .attn span { padding: 2px 0; border-radius: 2px; }
      .attn .cur { outline: 1px solid var(--saffron); }
    `,
  ],
})
export class AttentionView {
  readonly step = input<StepResult | null>(null);
  readonly info = input<ModelInfo | null>(null);

  readonly layer = signal(0);
  readonly head = signal(0);
  readonly average = signal(false);

  readonly layers = computed(() =>
    Array.from({ length: this.info()?.n_layer ?? 0 }, (_, i) => i),
  );
  readonly heads = computed(() => Array.from({ length: this.info()?.n_head ?? 0 }, (_, i) => i));

  /** Weights for the selected layer/head, or the head-average. */
  private readonly weights = computed<number[]>(() => {
    const a = this.step()?.attention;
    if (!a) return [];
    const layer = a[Math.min(this.layer(), a.length - 1)];
    if (!layer) return [];
    if (!this.average()) return layer[Math.min(this.head(), layer.length - 1)] ?? [];
    return layer[0].map((_, i) => layer.reduce((s, h) => s + h[i], 0) / layer.length);
  });

  readonly cells = computed(() => {
    const s = this.step();
    const w = this.weights();
    if (!s || !w.length) return [];

    // Cap what is rendered: a 256-character strip of coloured spans is both
    // unreadable and slow to paint.
    const SHOW = 220;
    const chars = s.context_chars;
    const start = Math.max(0, chars.length - SHOW);
    const max = Math.max(...w.slice(start));

    return chars.slice(start).map((c, k) => {
      const i = start + k;
      return {
        i,
        w: w[i] ?? 0,
        txt: c === '\n' ? '⏎' : c === ' ' ? '·' : c,
        bg: heat(w[i] ?? 0, max),
        last: i === chars.length - 1,
      };
    });
  });
}
