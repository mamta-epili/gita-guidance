import {
  Component,
  ChangeDetectionStrategy,
  OnDestroy,
  signal,
  output,
} from '@angular/core';

/**
 * Speech-to-text via the browser's Web Speech API.
 *
 * WHAT THIS COSTS, HONESTLY
 * Chrome does not recognise speech on your machine — it streams the audio to
 * Google's servers and streams text back. Everything else in this app runs
 * locally with no network calls at all, so the mic is the one component that
 * breaks that property. It is therefore opt-in per use (you must press it), and
 * the disclosure lives next to the button rather than in a privacy policy
 * nobody reads.
 *
 * SUPPORT
 *   Chrome / Edge   webkitSpeechRecognition, works
 *   Safari 14.1+    webkitSpeechRecognition, works, occasionally flaky
 *   Firefox         not implemented — the button hides itself entirely
 * Requires HTTPS or localhost, plus microphone permission.
 *
 * LANGUAGE
 * en-IN and hi-IN, because the corpus is indexed in both English and Hindi.
 * A Hindi question genuinely matches the Hindi rendering of a verse — the
 * per-language max-pooling in the retriever means the query never has to
 * declare which language it is in.
 */

type Recognition = any; // no lib.dom typing for SpeechRecognition

const LANGS = [
  { code: 'en-IN', label: 'EN' },
  { code: 'hi-IN', label: 'हि' },
];

@Component({
  selector: 'speech-input',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (supported()) {
      <!-- Right-aligned under the textarea: language first, mic in the corner. -->
      <div class="mic-row">
        <div class="langs" role="group" aria-label="Dictation language">
          @for (l of langs; track l.code) {
            <button
              type="button"
              class="lang"
              [class.sel]="lang() === l.code"
              (click)="lang.set(l.code)"
            >
              {{ l.label }}
            </button>
          }
        </div>

        <button
          type="button"
          class="mic"
          [class.live]="listening()"
          [attr.aria-pressed]="listening()"
          [attr.aria-label]="listening() ? 'Stop dictation' : 'Speak your question'"
          [title]="listening() ? 'Stop' : 'Speak your question'"
          (click)="toggle()"
        >
          <svg viewBox="0 0 24 24" width="17" height="17" fill="none" aria-hidden="true">
            <rect x="9" y="3" width="6" height="11" rx="3" fill="currentColor" />
            <path
              d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3"
              stroke="currentColor"
              stroke-width="1.8"
              stroke-linecap="round"
            />
          </svg>
        </button>
      </div>

      <!-- The disclosure stays on screen rather than becoming a tooltip: it is
           the one place this app touches the network. -->
      @if (listening()) {
        <p class="hint live-hint">listening…</p>
      } @else if (error()) {
        <p class="hint err">{{ error() }}</p>
      } @else {
        <p class="hint">
          speaking sends audio to the browser's speech service — the only part of this
          app that leaves your machine
        </p>
      }
    }
  `,
  styles: [
    `
      /* Column so the disclosure sits under the controls, both flush right. */
      :host {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
      }
      .mic-row {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .mic {
        width: 38px;
        height: 38px;
        flex: none;
        border-radius: 50%;
        padding: 0;
        display: grid;
        place-items: center;
        background: transparent;
        border: 1px solid var(--line);
        color: var(--dim);
        box-shadow: none;
        cursor: pointer;
      }
      .mic:hover {
        border-color: var(--peacock-deep);
        color: var(--peacock);
        filter: none;
        transform: none;
      }
      .mic:focus-visible { outline: 2px solid var(--peacock); outline-offset: 2px; }
      .mic.live {
        color: var(--vermillion);
        border-color: var(--vermillion);
        animation: pulse 1.5s ease-out infinite;
      }
      @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(196, 71, 47, 0.42); }
        100% { box-shadow: 0 0 0 12px rgba(196, 71, 47, 0); }
      }

      .langs { display: flex; gap: 3px; }
      .lang {
        background: transparent;
        border: 1px solid var(--line);
        color: var(--faint);
        border-radius: 5px;
        padding: 4px 9px;
        font: 11px var(--mono);
        box-shadow: none;
        font-weight: 500;
        letter-spacing: 0;
      }
      .lang:hover { color: var(--ink); filter: none; transform: none; }
      .lang.sel {
        color: var(--gold);
        border-color: rgba(233, 184, 80, 0.42);
        background: rgba(233, 184, 80, 0.07);
      }

      .hint {
        font-size: 11px;
        color: var(--faint);
        line-height: 1.45;
        max-width: 34ch;
        margin: 7px 0 0;
        text-align: right;
      }
      .hint.live-hint { color: var(--vermillion); font-family: var(--mono); }
      .hint.err { color: var(--saffron); }

      @media (prefers-reduced-motion: reduce) {
        .mic.live { animation: none; }
      }
    `,
  ],
})
export class SpeechInput implements OnDestroy {
  /** Fires continuously while speaking (interim), and once with the final text. */
  readonly transcript = output<string>();
  /** Fires with the final result only — the page uses this to auto-submit. */
  readonly finalTranscript = output<string>();

  readonly langs = LANGS;
  readonly lang = signal('en-IN');
  readonly listening = signal(false);
  readonly error = signal<string | null>(null);
  readonly supported = signal(false);

  private rec: Recognition | null = null;

  constructor() {
    if (typeof window === 'undefined') return;
    const Ctor =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    this.supported.set(!!Ctor);
  }

  toggle(): void {
    if (this.listening()) this.stop();
    else this.start();
  }

  private start(): void {
    const Ctor =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!Ctor) return;

    this.error.set(null);
    const rec: Recognition = new Ctor();
    rec.lang = this.lang();
    // One utterance at a time. `continuous` keeps the mic open indefinitely,
    // which for a search box means it stays hot while you read the results.
    rec.continuous = false;
    // Show words as they are recognised rather than after a pause — otherwise
    // it looks frozen for several seconds.
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    rec.onresult = (ev: any) => {
      let interim = '';
      let final = '';
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i];
        if (r.isFinal) final += r[0].transcript;
        else interim += r[0].transcript;
      }
      if (interim) this.transcript.emit(interim.trim());
      if (final) {
        const text = final.trim();
        this.transcript.emit(text);
        this.finalTranscript.emit(text);
      }
    };

    rec.onerror = (ev: any) => {
      // 'aborted' is what stop() produces; not worth reporting.
      const map: Record<string, string> = {
        'not-allowed': 'Microphone permission denied.',
        'service-not-allowed': 'Speech service unavailable.',
        'no-speech': "Didn't catch that — try again.",
        network: 'Speech service unreachable.',
      };
      if (ev.error !== 'aborted') this.error.set(map[ev.error] ?? `Mic error: ${ev.error}`);
      this.listening.set(false);
    };

    rec.onend = () => {
      this.listening.set(false);
      this.rec = null;
    };

    this.rec = rec;
    this.listening.set(true);
    try {
      rec.start();
    } catch {
      // start() throws if called while already running.
      this.listening.set(false);
    }
  }

  private stop(): void {
    try {
      this.rec?.stop();
    } catch {
      /* already stopped */
    }
    this.listening.set(false);
  }

  ngOnDestroy(): void {
    try {
      this.rec?.abort();
    } catch {
      /* nothing to abort */
    }
    this.rec = null;
  }
}
