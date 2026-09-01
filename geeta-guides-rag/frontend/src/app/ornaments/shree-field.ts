import { Component, ChangeDetectionStrategy, input, computed } from '@angular/core';

/**
 * श्री tiled across the whole viewport as wallpaper.
 *
 * Built from DOM text rather than a repeating background-image, deliberately:
 * a data-URI SVG containing Devanagari renders unreliably because the SVG has
 * no access to the page's font stack, and a raster tile would blur on retina
 * and need a licence-clean source. Real text in the page inherits --deva and
 * stays crisp at any zoom.
 *
 * Fixed position, so it behaves like wallpaper the content moves across rather
 * than texture glued to the document. Alternate rows are offset by half a cell,
 * which is what stops a tiled grid reading as a spreadsheet.
 *
 * aria-hidden, pointer-events: none, user-select: none — it must never be
 * announced, clicked or dragged into a copy-paste.
 */
@Component({
  selector: 'shree-field',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="field" aria-hidden="true">
      @for (r of rows(); track r) {
        <div class="line" [class.offset]="r % 2 === 1">
          @for (c of cols(); track c) {
            <span>श्री</span>
          }
        </div>
      }
    </div>
  `,
  styles: [
    `
      :host {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        user-select: none;
        overflow: hidden;
      }
      .field {
        /* Rotated slightly so it reads as woven cloth, not a table. The extra
           width/height and negative offset keep the corners covered after the
           rotation crops them. */
        position: absolute;
        top: -12%;
        left: -12%;
        width: 124%;
        height: 124%;
        transform: rotate(-8deg);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }
      .line {
        display: flex;
        justify-content: space-between;
        white-space: nowrap;
      }
      .line.offset { padding-left: 5.5%; }
      span {
        font-family: var(--deva);
        font-size: 30px;
        line-height: 1;
        /* Two very faint colours alternating by row would strobe; one flat
           value at very low alpha is calmer and keeps body text legible. */
        color: rgba(233, 184, 80, 0.055);
      }
      /* Fewer, larger glyphs on small screens — a dense grid on a phone reads
         as noise. */
      @media (max-width: 700px) {
        span { font-size: 24px; }
      }
    `,
  ],
})
export class ShreeField {
  /** Grid density. Generous defaults so large displays stay covered. */
  readonly columns = input(14);
  readonly linesCount = input(16);

  readonly cols = computed(() => Array.from({ length: this.columns() }, (_, i) => i));
  readonly rows = computed(() => Array.from({ length: this.linesCount() }, (_, i) => i));
}
