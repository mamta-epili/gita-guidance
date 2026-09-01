import {
  Component,
  ChangeDetectionStrategy,
  OnDestroy,
  signal,
  input,
} from '@angular/core';

const STORAGE_KEY = 'gita.flute.on';
const FADE_MS = 900;

/**
 * Ambient bansuri, with a mute toggle.
 *
 * THREE THINGS THAT MAKE BACKGROUND AUDIO GO WRONG, HANDLED HERE
 *
 * 1. Autoplay is blocked. Chrome and Safari refuse audible playback until the
 *    user has interacted with the origin, and `audio.play()` returns a rejected
 *    promise. Sites that appear to autoplay are usually benefiting from a high
 *    Media Engagement Index on a domain the visitor already knows. So we ATTEMPT
 *    playback, and when it is refused we sit quietly in the "off" state — the
 *    button then becomes the gesture that starts it. No console errors, no
 *    silent button that looks broken.
 *
 * 2. Unrequested audio is hostile. WCAG 1.4.2 requires a way to stop audio that
 *    plays longer than three seconds; this control is always visible and is the
 *    first thing in the tab order after the skip target. The choice persists in
 *    localStorage, so someone who muted it once is never asked again.
 *
 * 3. Hard cuts sound like a mistake. Volume ramps over ~900ms in both
 *    directions, which for ambient music is the difference between "music
 *    stopped" and "someone yanked a cable".
 *
 * If the audio file is absent the component hides itself entirely rather than
 * rendering a control that does nothing.
 */
@Component({
  selector: 'flute-audio',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (available()) {
      <button
        type="button"
        class="toggle"
        [class.on]="playing()"
        [attr.aria-pressed]="playing()"
        [attr.aria-label]="playing() ? 'Mute background flute' : 'Play background flute'"
        [title]="playing() ? 'Mute the flute' : 'Play the flute'"
        (click)="toggle()"
      >
        <!-- Equaliser bars: animated while playing, flat dots when muted. -->
        <span class="bars" aria-hidden="true">
          @for (b of bars; track b) {
            <i [style.animation-delay]="b * 0.18 + 's'"></i>
          }
        </span>
      </button>
    }
  `,
  styles: [
    `
      :host {
        position: fixed;
        left: 22px;
        bottom: 22px;
        z-index: 40;
      }
      .toggle {
        width: 54px;
        height: 54px;
        border-radius: 50%;
        padding: 0;
        display: grid;
        place-items: center;
        cursor: pointer;
        background: radial-gradient(circle at 34% 30%, #241a10, #100b06);
        border: 2px solid var(--gold-deep);
        box-shadow: 0 10px 26px -14px rgba(0, 0, 0, 0.9);
        transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.12s ease;
      }
      .toggle:hover { transform: translateY(-2px); border-color: var(--gold); filter: none; }
      .toggle:focus-visible {
        outline: 2px solid var(--peacock);
        outline-offset: 3px;
      }
      .toggle.on {
        border-color: var(--peacock);
        box-shadow: 0 0 0 4px rgba(35, 168, 154, 0.14),
          0 10px 26px -14px rgba(0, 0, 0, 0.9);
      }

      .bars { display: flex; align-items: center; gap: 3px; height: 20px; }
      .bars i {
        display: block;
        width: 3px;
        height: 4px;
        border-radius: 2px;
        background: var(--faint);
        transition: background 0.25s ease;
      }
      /* Muted: three flat dots, like the reference. Playing: bars dance. */
      .toggle.on .bars i {
        background: var(--peacock);
        animation: eq 1.05s ease-in-out infinite alternate;
      }
      @keyframes eq {
        from { height: 4px; }
        to { height: 18px; }
      }

      @media (prefers-reduced-motion: reduce) {
        .toggle.on .bars i { animation: none; height: 12px; }
      }
      @media (max-width: 700px) {
        :host { left: 14px; bottom: 14px; }
        .toggle { width: 46px; height: 46px; }
      }
    `,
  ],
})
export class FluteAudio implements OnDestroy {
  /** Path under public/. Absent file → the control hides itself. */
  readonly src = input('/audio/flute.mp3');
  /** Ceiling volume. Ambient means quiet; this is not a music player. */
  readonly maxVolume = input(0.32);

  readonly available = signal(false);
  readonly playing = signal(false);
  readonly bars = [0, 1, 2];

  private audio: HTMLAudioElement | null = null;
  private ramp: ReturnType<typeof setInterval> | null = null;

  constructor() {
    // SSR / prerender guard: no Audio constructor outside a browser.
    if (typeof window === 'undefined' || typeof Audio === 'undefined') return;

    const a = new Audio(this.src());
    a.loop = true;
    a.preload = 'auto';
    a.volume = 0;
    this.audio = a;

    // Only show the control once the file is known to exist and be playable.
    a.addEventListener('canplay', () => {
      this.available.set(true);
      if (this.wanted()) this.start();
    });
    a.addEventListener('error', () => {
      this.available.set(false);
      this.audio = null;
    });
  }

  /** Remembered preference; defaults to on, subject to autoplay policy. */
  private wanted(): boolean {
    try {
      const v = localStorage.getItem(STORAGE_KEY);
      return v === null ? true : v === '1';
    } catch {
      return true; // private browsing etc. — don't let storage break audio
    }
  }

  private remember(on: boolean): void {
    try {
      localStorage.setItem(STORAGE_KEY, on ? '1' : '0');
    } catch {
      /* storage unavailable; the session still works, it just won't persist */
    }
  }

  toggle(): void {
    if (this.playing()) {
      this.remember(false);
      this.stop();
    } else {
      this.remember(true);
      this.start();
    }
  }

  private start(): void {
    const a = this.audio;
    if (!a) return;
    a.play()
      .then(() => {
        this.playing.set(true);
        this.fadeTo(this.maxVolume());
      })
      .catch(() => {
        // Autoplay refused. Expected on a first visit — stay off and let the
        // button be the gesture. Deliberately not logged: it is not an error.
        this.playing.set(false);
      });
  }

  private stop(): void {
    const a = this.audio;
    if (!a) return;
    this.fadeTo(0, () => {
      a.pause();
      this.playing.set(false);
    });
  }

  /** Linear volume ramp over FADE_MS. */
  private fadeTo(target: number, done?: () => void): void {
    const a = this.audio;
    if (!a) return;
    if (this.ramp) clearInterval(this.ramp);

    const step = 24;
    const from = a.volume;
    const delta = target - from;
    let t = 0;

    this.ramp = setInterval(() => {
      t += step;
      const k = Math.min(1, t / FADE_MS);
      a.volume = Math.max(0, Math.min(1, from + delta * k));
      if (k >= 1) {
        if (this.ramp) clearInterval(this.ramp);
        this.ramp = null;
        done?.();
      }
    }, step);
  }

  ngOnDestroy(): void {
    if (this.ramp) clearInterval(this.ramp);
    this.audio?.pause();
    this.audio = null;
  }
}
