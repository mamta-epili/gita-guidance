import { Component, ChangeDetectionStrategy, input } from '@angular/core';

/**
 * Original inline SVG ornaments from Krishna's iconography.
 *
 * Drawn here rather than sourced as images for two reasons. First, licensing:
 * devotional art found online is almost always someone's copyright, and this
 * app is meant to be publishable. Second, SVG inherits the theme tokens, scales
 * to any size, and costs a few hundred bytes.
 *
 * Everything is aria-hidden — these carry no information a screen reader needs.
 */

/** Mor-pankh — the peacock feather in Krishna's crown. His signature. */
@Component({
  selector: 'orn-feather',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <svg
      [attr.width]="size()"
      [attr.height]="size() * 2.4"
      viewBox="0 0 50 120"
      fill="none"
      aria-hidden="true"
      [style.opacity]="opacity()"
      [style.transform]="'rotate(' + rotate() + 'deg)'"
    >
      <!-- shaft -->
      <path d="M25 120 C25 92 25 66 25 44" stroke="url(#shaft)" stroke-width="1.4" />

      <!-- barbs: two fans, drawn as short strokes off the shaft -->
      <g stroke="url(#barb)" stroke-width="0.85" opacity="0.75">
        @for (b of barbs; track b) {
          <path [attr.d]="'M25 ' + b + ' C 16 ' + (b - 5) + ', 9 ' + (b - 9) + ', 3 ' + (b - 16)" />
          <path [attr.d]="'M25 ' + b + ' C 34 ' + (b - 5) + ', 41 ' + (b - 9) + ', 47 ' + (b - 16)" />
        }
      </g>

      <!-- the eye: bronze halo, teal ring, indigo heart, dark pupil -->
      <ellipse cx="25" cy="30" rx="17" ry="24" fill="url(#halo)" opacity="0.5" />
      <ellipse cx="25" cy="29" rx="12.5" ry="17.5" fill="url(#teal)" />
      <ellipse cx="25" cy="28" rx="7.6" ry="11" fill="url(#indigo)" />
      <ellipse cx="25" cy="27" rx="3.6" ry="6" fill="#120c06" />
      <ellipse cx="23.6" cy="24.6" rx="1.15" ry="1.7" fill="#f6ead2" opacity="0.6" />

      <defs>
        <linearGradient id="shaft" x1="25" y1="120" x2="25" y2="40" gradientUnits="userSpaceOnUse">
          <stop stop-color="var(--gold-deep)" stop-opacity="0" />
          <stop offset="0.45" stop-color="var(--gold-deep)" stop-opacity="0.55" />
          <stop offset="1" stop-color="var(--gold)" />
        </linearGradient>
        <linearGradient id="barb" x1="25" y1="40" x2="3" y2="20" gradientUnits="userSpaceOnUse">
          <stop stop-color="var(--peacock)" />
          <stop offset="1" stop-color="var(--peacock-deep)" stop-opacity="0.15" />
        </linearGradient>
        <radialGradient id="halo"><stop stop-color="var(--gold)" /><stop offset="1" stop-color="var(--gold-deep)" stop-opacity="0.1" /></radialGradient>
        <radialGradient id="teal"><stop stop-color="var(--peacock)" /><stop offset="1" stop-color="var(--peacock-deep)" /></radialGradient>
        <!-- The one place blue appears: the jewel at the centre of the eye. -->
        <radialGradient id="indigo"><stop stop-color="var(--jewel)" /><stop offset="1" stop-color="#1a2547" /></radialGradient>
      </defs>
    </svg>
  `,
})
export class OrnFeather {
  readonly size = input(34);
  readonly rotate = input(0);
  readonly opacity = input(1);
  readonly barbs = [42, 47, 52, 57, 62, 67, 72, 77, 82, 88, 94, 100];
}

/** Bansuri — the flute, used as a section divider. */
@Component({
  selector: 'orn-flute',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <svg viewBox="0 0 420 26" width="100%" height="26" fill="none" aria-hidden="true">
      <!-- fading rules either side, so the flute reads as the centre of a rule -->
      <path d="M0 13 H128" stroke="url(#fadeL)" />
      <path d="M292 13 H420" stroke="url(#fadeR)" />

      <!-- body -->
      <rect x="132" y="9.5" width="156" height="7" rx="3.5" fill="url(#bamboo)" />
      <rect x="132" y="9.5" width="156" height="7" rx="3.5" stroke="var(--gold-deep)" stroke-opacity="0.5" />
      <!-- binding rings -->
      @for (x of rings; track x) {
        <rect [attr.x]="x" y="8" width="2" height="10" rx="1" fill="var(--gold)" opacity="0.55" />
      }
      <!-- finger holes -->
      @for (x of holes; track x) {
        <circle [attr.cx]="x" cy="13" r="1.5" fill="#05070f" opacity="0.85" />
      }
      <!-- blowing end -->
      <circle cx="290" cy="13" r="3.4" fill="none" stroke="var(--gold)" stroke-opacity="0.6" />

      <defs>
        <linearGradient id="bamboo" x1="132" y1="13" x2="288" y2="13" gradientUnits="userSpaceOnUse">
          <stop stop-color="var(--gold-deep)" /><stop offset="0.5" stop-color="var(--gold)" />
          <stop offset="1" stop-color="var(--gold-deep)" />
        </linearGradient>
        <linearGradient id="fadeL" x1="0" y1="13" x2="128" y2="13" gradientUnits="userSpaceOnUse">
          <stop stop-color="var(--line)" stop-opacity="0" /><stop offset="1" stop-color="var(--gold-deep)" stop-opacity="0.5" />
        </linearGradient>
        <linearGradient id="fadeR" x1="292" y1="13" x2="420" y2="13" gradientUnits="userSpaceOnUse">
          <stop stop-color="var(--gold-deep)" stop-opacity="0.5" /><stop offset="1" stop-color="var(--line)" stop-opacity="0" />
        </linearGradient>
      </defs>
    </svg>
  `,
  styles: [`:host { display: block; margin: 30px 0; }`],
})
export class OrnFlute {
  readonly rings = [136, 174, 212, 250, 282];
  readonly holes = [154, 168, 182, 196, 210, 224, 238];
}

/** Padma — lotus, marking a verse. */
@Component({
  selector: 'orn-lotus',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <svg [attr.width]="size()" [attr.height]="size()" viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <g stroke="var(--peacock)" stroke-opacity="0.75" stroke-width="1.1" fill="none">
        <path d="M16 27 C16 19 16 13 16 6" />
        <path d="M16 27 C10 21 7 17 6 11" />
        <path d="M16 27 C22 21 25 17 26 11" />
        <path d="M16 27 C11 24 8 22 4 19" />
        <path d="M16 27 C21 24 24 22 28 19" />
      </g>
      <circle cx="16" cy="27" r="2.1" fill="var(--gold)" opacity="0.9" />
    </svg>
  `,
})
export class OrnLotus {
  readonly size = input(22);
}

/** Sudarshana chakra — the discus, used as a quiet loading spinner. */
@Component({
  selector: 'orn-chakra',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <svg
      [attr.width]="size()" [attr.height]="size()" viewBox="0 0 40 40" fill="none"
      aria-hidden="true" [class.spin]="spin()"
    >
      <circle cx="20" cy="20" r="17.5" stroke="var(--gold)" stroke-opacity="0.5" />
      <circle cx="20" cy="20" r="11" stroke="var(--peacock)" stroke-opacity="0.65" />
      <circle cx="20" cy="20" r="3.2" fill="var(--gold)" opacity="0.85" />
      @for (a of spokes; track a) {
        <path
          d="M20 6 L20 34" stroke="var(--gold)" stroke-opacity="0.32" stroke-width="0.9"
          [attr.transform]="'rotate(' + a + ' 20 20)'"
        />
      }
    </svg>
  `,
  styles: [
    `
      .spin { animation: rot 2.6s linear infinite; transform-origin: 50% 50%; }
      @keyframes rot { to { transform: rotate(360deg); } }
    `,
  ],
})
export class OrnChakra {
  readonly size = input(26);
  readonly spin = input(false);
  readonly spokes = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5];
}
