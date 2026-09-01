import { Component, ChangeDetectionStrategy } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { ShreeField } from './ornaments/shree-field';
import { FluteAudio } from './ornaments/flute-audio';

/**
 * Shell. The श्री wallpaper and the flute control live here rather than on a
 * page, so each is a single instance shared by every route — the audio in
 * particular must not restart when you navigate between / and /lab.
 */
@Component({
  selector: 'app-root',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterOutlet, ShreeField, FluteAudio],
  template: `
    <shree-field />
    <router-outlet />
    <flute-audio />
  `,
})
export class App {}
