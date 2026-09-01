import { Component, ChangeDetectionStrategy, input } from '@angular/core';
import { VerseHit } from '../core/api';
import { OrnLotus } from '../ornaments/ornaments';

/**
 * One retrieved verse, in the display order fixed by PRD FR-3.5:
 * Devanagari shloka → IAST → English → Hindi → citation.
 *
 * The shloka comes first because an answer that opens with a paraphrase asks
 * the reader to trust the paraphrase; one that opens with the verse asks them
 * to check it. Nothing here is generated — every string is a verbatim field
 * from data/verses.json.
 *
 * Visually the card is a palm-leaf page: warm ground, gold left edge, the
 * Devanagari given the space it deserves rather than being treated as a
 * subtitle to the English.
 */
@Component({
  selector: 'verse-card',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [OrnLotus],
  template: `
    @if (verse(); as v) {
      <article class="card" [class.muted]="muted()">
        <header>
          <span class="cite">
            <orn-lotus [size]="18" />
            {{ v.citation }}
            <em class="who-said">{{ v.speaker_name }}</em>
          </span>
          @if (v.score !== null) {
            <span
              class="score"
              [title]="'cosine similarity · matched the ' + v.matched_lang + ' rendering'"
            >
              {{ v.score.toFixed(3) }}
              @if (v.matched_lang) {
                <em>{{ v.matched_lang }}</em>
              }
            </span>
          }
        </header>

        @if (v.sanskrit) {
          <p class="deva">{{ v.sanskrit }}</p>
        }
        @if (v.iast) {
          <p class="iast">{{ v.iast }}</p>
        }

        @if (v.english || v.hindi) {
          <hr class="rule" />
        }

        @if (v.english; as en) {
          <div class="rendering">
            <span class="who">English · {{ en.translator }}</span>
            <p>{{ en.text }}</p>
          </div>
        }
        @if (v.hindi; as hi) {
          <div class="rendering">
            <span class="who">
              हिन्दी · {{ hi.translator }}
              @if (!hi.redistributable) {
                <b
                  class="restricted"
                  title="Copyrighted. Fine on your machine; must not be published."
                  >local only</b
                >
              }
            </span>
            <p class="hi">{{ hi.text }}</p>
          </div>
        }
      </article>
    }
  `,
  styles: [
    `
      .card {
        position: relative;
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--bg-2) 74%);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 22px 26px 24px;
        margin-bottom: 18px;
        overflow: hidden;
        box-shadow: 0 22px 44px -30px rgba(0, 0, 0, 0.95),
          inset 0 1px 0 rgba(255, 255, 255, 0.035);
      }
      /* gold leading edge, like a gilded manuscript margin */
      .card::before {
        content: '';
        position: absolute;
        inset: 0 auto 0 0;
        width: 2px;
        background: linear-gradient(
          180deg,
          transparent,
          var(--gold) 18%,
          var(--peacock) 82%,
          transparent
        );
        opacity: 0.75;
      }

      header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
      }
      .cite {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font: 11px var(--mono);
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--gold);
      }
      .score { font: 11px var(--mono); color: var(--faint); white-space: nowrap; }
      .score em { color: var(--peacock); font-style: normal; margin-left: 6px; }

      .who-said {
        font: italic 11px var(--serif);
        color: var(--faint);
        text-transform: none;
        letter-spacing: 0;
        margin-left: 2px;
      }

      /* Muted variant: smaller Devanagari, cooler edge, less presence. Still
         fully readable — this is de-emphasis, not a disabled state. */
      .card.muted { padding: 18px 22px 20px; background: linear-gradient(160deg, var(--panel) 0%, var(--bg-2) 74%); }
      .card.muted::before {
        background: linear-gradient(180deg, transparent, var(--peacock-deep) 30%, transparent);
        opacity: 0.5;
      }
      .card.muted .deva { font-size: 17px; line-height: 1.95; color: var(--dim); text-shadow: none; }
      .card.muted .cite { color: var(--peacock); }
      .card.muted .rendering p { font-size: 14.5px; color: var(--dim); }

      .deva {
        font-family: var(--deva);
        font-size: 21px;
        line-height: 2.05;
        color: var(--gold);
        margin: 18px 0 0;
        white-space: pre-line;
        text-shadow: 0 0 34px rgba(233, 184, 80, 0.14);
      }
      .iast {
        font: 12.5px/1.9 var(--mono);
        color: var(--faint);
        margin: 10px 0 0;
        white-space: pre-line;
        letter-spacing: 0.01em;
      }

      hr.rule { margin: 20px 0 4px; }

      .rendering { margin-top: 16px; }
      .who {
        font: 10px var(--mono);
        color: var(--peacock);
        text-transform: uppercase;
        letter-spacing: 0.16em;
      }
      .restricted {
        color: var(--saffron);
        font-weight: 600;
        margin-left: 9px;
        text-transform: none;
        letter-spacing: 0;
        border: 1px solid rgba(232, 131, 60, 0.35);
        border-radius: 3px;
        padding: 1px 5px;
      }
      .rendering p {
        margin: 7px 0 0;
        line-height: 1.75;
        font-family: var(--serif);
        font-size: 15.5px;
      }
      /* 0.9 of the English above it. Devanagari at an equal px reads larger —
         bigger x-height, and the shirorekha closes the top of every letter, so
         the line carries more ink. Matching the numbers makes the two
         translations look unequal in weight. */
      .rendering p.hi { font-family: var(--deva); font-size: 14px; line-height: 1.9; }
      .card.muted .rendering p.hi { font-size: 13px; }
    `,
  ],
})
export class VerseCard {
  readonly verse = input<VerseHit | null>(null);
  /** Recede visually — used for the Arjuna/Sanjaya block. */
  readonly muted = input(false);
}
