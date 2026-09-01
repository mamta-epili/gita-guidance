import { Component, ChangeDetectionStrategy, inject, input, signal, OnDestroy } from '@angular/core';
import { Api } from '../core/api';

/**
 * Streamed generation, one character per forward pass.
 *
 * The streaming is not decoration. One full pass over the whole context yields
 * exactly one character; then the context grows by one and it all runs again.
 * Watching 240 characters arrive is watching 240 forward passes.
 */
@Component({
  selector: 'generate-stream',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="row" style="margin-top:0">
      <button (click)="start()" [disabled]="running()">Generate {{ count }} characters</button>
      <button class="ghost" (click)="stop()" [disabled]="!running()">Stop</button>
      <label>from the prompt above</label>
    </div>

    <pre class="out"
      ><span class="seed">{{ seed() }}</span>{{ produced()
      }}@if (running()) {<span class="cursor">&nbsp;</span>}</pre
    >
  `,
  styles: [
    `
      .out {
        background: #0c0a09; border: 1px solid var(--line); border-radius: 4px;
        padding: 14px; font: 13px/1.75 var(--mono); white-space: pre-wrap;
        word-break: break-word; min-height: 150px; max-height: 340px;
        overflow: auto; margin: 12px 0 0; color: var(--ink);
      }
      .seed { color: var(--faint); }
      .cursor {
        display: inline-block; width: 7px; background: var(--saffron);
        animation: blink 1s steps(2) infinite;
      }
      @keyframes blink { 50% { opacity: 0; } }
    `,
  ],
})
export class GenerateStream implements OnDestroy {
  private api = inject(Api);

  readonly prompt = input<string>('');
  readonly temperature = input<number>(1);
  readonly count = 240;

  readonly running = signal(false);
  readonly seed = signal('');
  readonly produced = signal('');

  private cancel: (() => void) | null = null;

  start(): void {
    this.stop();
    this.seed.set(this.prompt());
    this.produced.set('');
    this.running.set(true);

    // Signals update the view even though EventSource callbacks fire outside
    // Angular's zone — no NgZone.run needed.
    this.cancel = this.api.stream(
      this.prompt(),
      this.count,
      this.temperature(),
      (ch) => this.produced.update((s) => s + ch),
      () => {
        this.running.set(false);
        this.cancel = null;
      },
    );
  }

  stop(): void {
    this.cancel?.();
    this.cancel = null;
    this.running.set(false);
  }

  ngOnDestroy(): void {
    this.stop();
  }
}
