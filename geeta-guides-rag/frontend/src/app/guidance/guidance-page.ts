import { Component, ChangeDetectionStrategy, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Api } from '../core/api';
import { VerseCard } from './verse-card';
import { SpeechInput } from './speech-input';
import { OrnFeather, OrnFlute, OrnChakra } from '../ornaments/ornaments';

/**
 * The feeling chips, each paired with the Sanskrit term the Gita itself uses.
 * These aren't decorative translations — every one is a word that appears in
 * the text, which is why they read as belonging here:
 *
 *   चिन्ता  chintā   anxiety
 *   भय      bhaya    fear          (2:56, "loosed from passion, fear and anger")
 *   क्रोध    krodha   anger         (2:62-63, the desire→anger→delusion chain)
 *   शोक     śoka     grief         (2:11, "thou grievest for those that should not be grieved for")
 *   मोह     moha     delusion      (2:63, "from anger proceedeth delusion")
 *   संशय    saṃśaya  doubt         (4:40, "he who doubteth")
 */
const CHIPS = [
  { en: 'stressed', sa: 'चिन्ता' },
  { en: 'afraid', sa: 'भय' },
  { en: 'angry', sa: 'क्रोध' },
  { en: 'grieving', sa: 'शोक' },
  { en: 'confused', sa: 'मोह' },
  { en: 'lost', sa: 'संशय' },
];

@Component({
  selector: 'guidance-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [VerseCard, SpeechInput, RouterLink, OrnFeather, OrnFlute, OrnChakra],
  template: `
    <div class="wrap">
      <!-- Mangala-shloka. Traditionally opens a recitation of the Gita, so it
           opens the page. -->
      <section class="invocation">
        <!-- Each half in its own span rather than separated by <br>, so the
             shloka sits on one line and only breaks at the danda when the
             viewport is too narrow for it to stay readable. -->
        <p class="deva">
          <span>कृष्णाय वासुदेवाय हरये परमात्मने ।</span>
          <span>प्रणतक्लेशनाशाय गोविन्दाय नमो नमः ॥</span>
        </p>
        <p class="iast">
          <span>kṛṣṇāya vāsudevāya haraye paramātmane |</span>
          <span>praṇata-kleśanāśāya govindāya namo namaḥ ||</span>
        </p>
        <p class="gloss">
          To Krishna, son of Vasudeva, to Hari, the Supreme Self; to Govinda, destroyer of
          the afflictions of those who bow to him — salutations, again and again.
        </p>
      </section>

      <orn-flute />

      <header>
        <div class="masthead">
          <orn-feather [size]="40" [rotate]="-13" />
          <div>
            <h1><span class="om">ॐ</span>Gita guidance</h1>
            <p class="tagline">पार्थाय प्रतिबोधिताम् — spoken to Arjuna, for anyone</p>
          </div>
        </div>
        <p class="sub">
          Describe a situation or a feeling — the shlokas that speak to it, quoted with
          chapter and verse.
        </p>
      </header>

      <div class="panel ask">
        <textarea
          rows="2"
          spellcheck="false"
          autocorrect="off"
          data-gramm="false"
          placeholder="What weighs on you?"
          [value]="q()"
          (input)="q.set($any($event.target).value)"
          (keydown.enter)="onEnter($event)"
        ></textarea>

        <!-- Chips left, dictation right, one row. -->
        <div class="chips-row">
          <div class="chips">
            @for (c of chips; track c.en) {
              <button class="chip-btn" (click)="pick(c.en)">
                <span class="sa">{{ c.sa }}</span>
                <span class="en">{{ c.en }}</span>
              </button>
            }
          </div>

          <speech-input
            (transcript)="q.set($event)"
            (finalTranscript)="onDictated($event)"
          />
        </div>

        <div class="row">
          <button (click)="submit()" [disabled]="api.asking() || !q().trim()">
            {{ api.asking() ? 'Seeking…' : 'Find shlokas' }}
          </button>
          @if (api.asking()) {
            <orn-chakra [size]="24" [spin]="true" />
          }
          <label>Enter to submit · Shift+Enter for a new line</label>
        </div>
      </div>

      <!-- One diagnostic, not two. A failed /ready is the more specific
           explanation, so it wins. -->
      @if (api.ready(); as r) {
        @if (!r.ready) {
          <div class="panel problem">
            <p><strong>Retrieval isn't available yet.</strong></p>
            <pre>{{ r.reason }}</pre>
            <p class="note" style="margin:10px 0 0">
              If the backend is running, its dependencies or the embedding index are
              missing — <code>pip install -r requirements.txt</code> in
              <code>geeta-guides-rag</code>, then <code>make embed</code> in
              <code>geeta-guides</code>.
            </p>
          </div>
        } @else if (api.error(); as err) {
          <div class="panel problem"><pre>{{ err }}</pre></div>
        }
      }

      @if (api.guidance(); as g) {
        @if (g.verses.length) {
          <orn-flute />

          <!-- Krishna's answer. Enclosed in its own lit panel so it reads as a
               distinct thing rather than the first few items of one long list. -->
          @if (g.teaching.length) {
            <section class="teaching">
              <div class="teaching-head">
                <orn-feather [size]="30" [rotate]="-10" />
                <div class="titles">
                  <span class="sa-head">श्रीभगवानुवाच</span>
                  <span class="say">The Blessed Lord said</span>
                </div>
                <span class="badge">the answer</span>
              </div>
              @for (v of g.teaching; track v.id) {
                <verse-card [verse]="v" />
              }
            </section>
          }

          <!-- Arjuna's and Sanjaya's verses: your own question reflected back.
               Open by default. The separation is carried by the answer block's
               lit enclosure and the flute divider, not by hiding this. The
               toggle stays so it can be collapsed when it's in the way. -->
          @if (g.dialogue.length) {
            <orn-flute />
            <section class="dialogue">
              <button
                type="button"
                class="disclose"
                [attr.aria-expanded]="showDialogue()"
                (click)="showDialogue.set(!showDialogue())"
              >
                <span class="caret" aria-hidden="true">{{ showDialogue() ? '−' : '+' }}</span>
                <span class="sa-head">अर्जुन उवाच</span>
                <span class="sep" aria-hidden="true">·</span>
                <span class="say"
                  >the same difficulty, as Arjuna put it ({{ g.dialogue.length }})</span
                >
              </button>

              @if (showDialogue()) {
                <p class="note">
                  These matched your words most closely — because Arjuna is describing the
                  same trouble. They are the question, not the teaching.
                </p>
                @for (v of g.dialogue; track v.id) {
                  <verse-card [verse]="v" [muted]="true" />
                }
              }
            </section>
          }

          <p class="note timing">
            {{ g.verses.length }} of {{ g.pool_size }} retrieved · {{ g.ms }} ms ·
            {{ g.score_note }}
          </p>
        } @else {
          <p class="note">{{ g.note || 'Nothing found.' }}</p>
        }
      }

      <orn-flute />

      <footer>
        <div class="feathers" aria-hidden="true">
          <orn-feather [size]="20" [rotate]="-20" [opacity]="0.4" />
          <orn-feather [size]="26" [rotate]="0" [opacity]="0.55" />
          <orn-feather [size]="20" [rotate]="20" [opacity]="0.4" />
        </div>
        <p>
          Retrieval only. The shloka, its translations and its citation are verbatim from
          the corpus — nothing on this path writes text, so nothing on it can invent
          scripture.
        </p>
        <p>
          <a routerLink="/lab">See the character-level GPT this project began with →</a>
        </p>
        <p class="shanti">ॐ शान्तिः शान्तिः शान्तिः</p>
      </footer>
    </div>
  `,
  styles: [
    `
      /* --- mangala-shloka ------------------------------------------------ */
      .invocation { text-align: center; margin: 6px 0 0; }

      /* Flex + nowrap keeps both halves on one line; the font size scales with
         the viewport so it fits rather than wrapping. Capped at 21px because
         .wrap stops growing at 1000px, and floored so it never becomes
         unreadable — below that floor the halves wrap instead (see the media
         query), which is the traditional two-line setting anyway. */
      .invocation .deva,
      .invocation .iast {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        gap: 0.45em;
        white-space: nowrap;
      }
      .invocation .deva {
        font-family: var(--deva);
        font-size: clamp(13px, 2.02vw, 21px);
        line-height: 2.05;
        color: var(--gold);
        margin: 0;
        text-shadow: 0 0 40px rgba(233, 184, 80, 0.18);
      }
      .invocation .iast {
        font-family: var(--mono);
        font-size: clamp(9px, 1.12vw, 11.5px);
        line-height: 1.9;
        color: var(--peacock);
        margin: 12px 0 0;
        letter-spacing: 0.02em;
        opacity: 0.85;
      }

      /* Too narrow for one readable line — break at the danda. */
      @media (max-width: 620px) {
        .invocation .deva,
        .invocation .iast {
          flex-wrap: wrap;
          gap: 0;
        }
        .invocation .deva span,
        .invocation .iast span { flex: 0 0 100%; }
        .invocation .deva { font-size: 17px; }
        .invocation .iast { font-size: 10.5px; }
      }
      .invocation .gloss {
        font-family: var(--serif);
        font-style: italic;
        font-size: 13.5px;
        line-height: 1.7;
        color: var(--faint);
        max-width: 62ch;
        margin: 14px auto 0;
      }
      header {
        position: relative;
        z-index: 1;
        border-bottom: 1px solid var(--line-soft);
        padding-bottom: 24px;
        margin-bottom: 28px;
      }
      /* The intro is short enough to sit on one line at this width; the 62ch
         cap from styles.css would wrap it and leave dead space on the right. */
      header .sub { max-width: none; }
      .masthead { display: flex; align-items: flex-start; gap: 6px; }
      .tagline {
        margin: 2px 0 14px;
        font-family: var(--deva);
        font-size: 13.5px;
        color: var(--peacock);
        letter-spacing: 0.02em;
      }

      .ask textarea { min-height: 62px; font-size: 16px; }

      /* flex-start rather than center: the dictation block is taller than the
         chips because of the disclosure line beneath it, and centring would
         push the chips down to match. */
      .chips-row {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
        margin-top: 14px;
        flex-wrap: wrap;
      }
      .chips-row speech-input { flex: 0 0 auto; }

      .chips { display: flex; flex-wrap: wrap; gap: 7px; flex: 1 1 320px; }
      .chip-btn {
        display: inline-flex;
        align-items: baseline;
        gap: 7px;
        background: rgba(35, 168, 154, 0.06);
        border: 1px solid rgba(35, 168, 154, 0.28);
        border-radius: 999px;
        padding: 5px 15px 6px;
        cursor: pointer;
        box-shadow: none;
        letter-spacing: 0;
        font-weight: 400;
      }
      .chip-btn .sa {
        font-family: var(--deva);
        font-size: 14px;
        color: var(--gold);
        line-height: 1.4;
      }
      .chip-btn .en {
        font: italic 12.5px var(--serif);
        color: var(--peacock);
      }
      .chip-btn:hover {
        background: rgba(233, 184, 80, 0.09);
        border-color: rgba(233, 184, 80, 0.42);
        filter: none;
        transform: none;
      }
      .chip-btn:hover .en { color: var(--gold); }

      h2 {
        display: flex; align-items: baseline; gap: 12px; margin-bottom: 18px;
        flex-wrap: wrap;
      }
      .sa-head {
        font-family: var(--deva);
        font-size: 17px;
        color: var(--gold);
        text-transform: none;
        letter-spacing: 0;
      }
      .say {
        font: italic 13px var(--serif);
        color: var(--dim);
        text-transform: none;
        letter-spacing: 0;
      }

      /* ---- the answer -----------------------------------------------------
         Its own lit enclosure: gold hairline, a warm wash falling from the top
         edge, and a glow beneath. The cards inside sit on this surface, which is
         what stops the two blocks reading as one continuous list. */
      .teaching {
        margin-top: 6px;
        padding: 20px 20px 4px;
        border: 1px solid rgba(233, 184, 80, 0.3);
        border-radius: 12px;
        background:
          radial-gradient(720px 220px at 50% -12%, rgba(233, 184, 80, 0.11), transparent 70%),
          linear-gradient(180deg, rgba(233, 184, 80, 0.05), rgba(233, 184, 80, 0.012));
        box-shadow: 0 34px 70px -50px rgba(233, 184, 80, 0.3);
      }
      .teaching-head {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 16px;
        margin-bottom: 18px;
        border-bottom: 1px solid rgba(233, 184, 80, 0.22);
      }
      .teaching-head .titles { display: flex; flex-direction: column; gap: 2px; flex: 1; }
      .teaching-head .sa-head { font-size: 19px; }
      .badge {
        font: 10px var(--mono);
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #241703;
        background: linear-gradient(180deg, var(--gold), var(--gold-deep));
        border-radius: 999px;
        padding: 4px 11px;
        white-space: nowrap;
      }
      /* Slightly darker than the enclosure, so each verse reads as a card
         resting on a lit surface rather than merging into it. */
      .teaching verse-card .card { background: linear-gradient(160deg, #1d1509 0%, #120d07 74%); }

      /* ---- the question, alongside ---------------------------------------- */
      .dialogue { margin-top: 8px; }
      .disclose {
        display: flex;
        align-items: baseline;
        gap: 10px;
        width: 100%;
        text-align: left;
        background: transparent;
        border: 1px dashed rgba(35, 168, 154, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: none;
        cursor: pointer;
        font-weight: 400;
        letter-spacing: 0;
      }
      .disclose:hover {
        background: rgba(35, 168, 154, 0.05);
        border-color: rgba(35, 168, 154, 0.5);
        filter: none;
        transform: none;
      }
      .disclose .caret {
        font: 15px var(--mono); color: var(--peacock); width: 12px; flex: none;
      }
      .disclose .sa-head { color: var(--peacock); font-size: 15px; }
      .disclose .sep { color: var(--line); }
      .disclose .say { color: var(--faint); font-size: 12.5px; }
      .dialogue > .note { margin: 18px 0; }
      .timing { margin-top: 26px; }
      .meta {
        font: 10px var(--mono); color: var(--faint);
        text-transform: none; letter-spacing: 0;
      }

      .problem {
        border-left: 2px solid var(--vermillion);
        margin-top: 20px;
      }
      .problem pre {
        margin: 8px 0 0; font: 12px/1.6 var(--mono); color: var(--dim);
        white-space: pre-wrap; word-break: break-word;
      }
      code { font: 12px var(--mono); color: var(--gold); }

      footer { color: var(--faint); font-size: 13px; text-align: center; }
      .feathers {
        display: flex; justify-content: center; align-items: flex-end;
        gap: 10px; margin-bottom: 14px;
      }
      footer p { max-width: 62ch; margin: 8px auto; }
      .shanti {
        font-family: var(--deva);
        font-size: 15px;
        color: var(--gold);
        opacity: 0.55;
        letter-spacing: 0.08em;
        margin-top: 22px !important;
      }
    `,
  ],
})
export class GuidancePage {
  readonly api = inject(Api);
  readonly chips = CHIPS;
  readonly q = signal('');
  /** Arjuna's block is open by default; the toggle is there to collapse it. */
  readonly showDialogue = signal(true);

  constructor() {
    this.api.checkReady();
  }

  onEnter(ev: Event): void {
    const e = ev as KeyboardEvent;
    // Enter submits; Shift+Enter inserts a newline, as people expect from a
    // chat-shaped box.
    if (!e.shiftKey) {
      e.preventDefault();
      this.submit();
    }
  }

  /** Dictation finished. Search straight away — having spoken the question,
      being made to then click a button is an odd extra step. */
  onDictated(text: string): void {
    this.q.set(text);
    this.submit();
  }

  pick(word: string): void {
    this.q.set(`I am feeling ${word}. Guide me.`);
    this.submit();
  }

  submit(): void {
    const text = this.q().trim();
    if (!text) return;
    // Reopen on a new question, so a collapse is per-result rather than sticky.
    this.showDialogue.set(true);
    this.api.ask(text, 5);
  }
}
