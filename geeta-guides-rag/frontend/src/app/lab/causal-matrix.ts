import { Component, ChangeDetectionStrategy, input, computed } from '@angular/core';
import { StepResult, heat } from '../core/api';

/**
 * The full attention matrix of layer 0 head 0, as a heatmap.
 *
 * Row t is what position t attended to. The upper triangle is empty because
 * masked_fill(tril == 0, -inf) zeroed it BEFORE the softmax — a position cannot
 * see its own future. This is the clearest single picture of what the causal
 * mask does, which is why it gets its own component.
 */
@Component({
  selector: 'causal-matrix',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (rows().length) {
      <div class="matrix" [style.gridTemplateColumns]="cols()">
        @for (row of rows(); track row.r) {
          @for (cell of row.cells; track cell.c) {
            <div
              [style.background]="cell.bg"
              [title]="'pos ' + row.r + ' → pos ' + cell.c + ': ' + cell.pct + '%'"
            ></div>
          }
        }
      </div>
      <p class="note" style="margin:12px 0 0">
        {{ rows().length }}×{{ rows().length }} positions. Hover any cell for its exact
        weight. Every row sums to 1.
      </p>
    } @else {
      <p class="note">Type at least a few characters above.</p>
    }
  `,
  styles: [
    `
      .matrix { display: grid; gap: 1px; margin-top: 6px; }
      .matrix div { width: 100%; aspect-ratio: 1; }
    `,
  ],
})
export class CausalMatrix {
  readonly step = input<StepResult | null>(null);

  private readonly matrix = computed(() => this.step()?.attention_matrix_l0h0 ?? []);

  readonly cols = computed(() => `repeat(${this.matrix().length}, 1fr)`);

  readonly rows = computed(() => {
    const m = this.matrix();
    if (!m.length) return [];
    const max = Math.max(...m.flat());
    return m.map((row, r) => ({
      r,
      cells: row.map((v, c) => ({
        c,
        // Cells above the diagonal are painted flat black rather than
        // heat(0) — the point is that they are structurally absent, not
        // merely small.
        bg: c <= r ? heat(v, max) : '#0a0908',
        pct: (v * 100).toFixed(2),
      })),
    }));
  });
}
