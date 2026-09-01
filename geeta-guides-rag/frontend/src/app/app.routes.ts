import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    // The product. Retrieval-backed verse recommendation.
    path: '',
    loadComponent: () => import('./guidance/guidance-page').then((m) => m.GuidancePage),
  },
  {
    // The foundations. A demo of the char-level GPT, deliberately separate from
    // the product so the app never implies that model is answering anything.
    path: 'lab',
    loadComponent: () => import('./lab/lab-page').then((m) => m.LabPage),
  },
  { path: '**', redirectTo: '' },
];
