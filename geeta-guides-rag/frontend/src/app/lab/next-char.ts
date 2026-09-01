import { Component, ChangeDetectionStrategy, input, computed } from '@angular/core';
import { StepResult, label } from '../core/api';

/**
 * The model's next-character distribution.
 *
 * This is the model's entire output. Generation is only: sample from this,
 * append, run the forward pass again. Worth stating in the UI, because it is
 * the single most commonly misunderstood thing about how these models work.
 */
@Component({
  selector: 'next-char',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (step(); as s) {
      <div class="bars">
        @for (t of s.top; track t.char) {
          <div class="bar">
            <div class="ch">{{ lbl(t.char) }}</div>
            <div class="track">
              <div class="fill" [style.width.%]="pct(t.prob)"></div>
            </div>
            <div class="pc">{{ (t.prob * 100).toFixed(1) }}%</div>
          </div>
        }
      </div>
    } @else {
      <p class="note">Type above to run a forward pass.</p>
    }
  `,
  styles: [
    `
      .bar {
        display: grid; grid-template-columns: 34px 1fr 60px;
        gap: 9px; align-items: center; margin-bottom: 5px;
      }
      .ch {
        font: 13px var(--mono); color: var(--gold); text-align: center;
        background: #0c0a09; border: 1px solid var(--line);
        border-radius: 3px; padding: 2px 0;
      }
      .track { height: 15px; background: #0c0a09; border-radius: 2px; overflow: hidden; }
      .fill { height: 100%; background: linear-gradient(90deg, var(--deep), var(--gold)); }
      .pc { font: 11px var(--mono); color: var(--dim); text-align: right; }
    `,
  ],
})
export class NextChar {
  readonly step = input<StepResult | null>(null);

  private readonly max = computed(() => {
    const s = this.step();
    return s?.top.length ? s.top[0].prob : 1;
  });

  lbl = label;
  pct(p: number): number {
    return (p / this.max()) * 100;
  }
}
