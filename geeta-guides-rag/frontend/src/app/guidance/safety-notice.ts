/**
 * Superseded — kept as a stub so no stale import breaks the build.
 *
 * This used to render a crisis notice that blocked verses. The design changed:
 * nothing is blocked now, and questions about despair get a hand-chosen set of
 * verses instead of the retriever's top-k. See app/safety.py and
 * <verse-card [asks]>.
 *
 * Safe to delete once you have confirmed nothing imports it.
 */
import { Component, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'safety-notice',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: ``,
})
export class SafetyNotice {}
