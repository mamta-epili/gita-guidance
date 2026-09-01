# Background flute audio

Drop a looping bansuri track here as **`flute.mp3`**. The control at the bottom
left appears automatically once the file is present and playable, and hides
itself if it is missing — no broken button.

```
frontend/public/audio/flute.mp3
```

Angular copies everything in `public/` into the build, so it is served at
`/audio/flute.mp3`. To use a different name or format, pass it in:

```html
<flute-audio src="/audio/my-track.ogg" [maxVolume]="0.25" />
```

## Where to get a track you can actually use

**Not from YouTube.** Downloading audio from YouTube breaks its Terms of
Service, and the recordings there are almost always someone's copyright —
including the ones that look like anonymous devotional uploads. Using one in a
deployed app is a real exposure, and "it was on YouTube" is not a licence.

These are legitimately free, and all have flute/bansuri material:

| Source | Licence | Notes |
|---|---|---|
| [Pixabay Audio](https://pixabay.com/music/) | Pixabay Content Licence — free for commercial use, no attribution required | Easiest. Search "bansuri", "bamboo flute", "indian flute" |
| [Free Music Archive](https://freemusicarchive.org/) | Per-track CC — check each | Filter by licence; some are CC0 |
| [Freesound](https://freesound.org/) | Per-sound CC — check each | More raw recordings than compositions; good for a single sustained phrase to loop |
| [YouTube Audio Library](https://studio.youtube.com/) | Free to use, some need attribution | Confusingly, this *is* fine — it is YouTube's own licensed library, not scraped uploads |
| [Kevin MacLeod / Incompetech](https://incompetech.com/music/) | CC-BY — attribution required | Long-standing, reliable |

Whichever you pick, **record the source and licence** — the same discipline
`data/verses.json` applies to the translations. If attribution is required, the
place for it is the footer.

## What makes a good loop here

- **Sustained and sparse.** This plays under text someone is reading. Anything
  with a melody that resolves will pull attention every eight bars.
- **Seamless.** Trim to a zero crossing at both ends, or the loop clicks.
  `loop = true` gives no crossfade.
- **Small.** Aim under ~2 MB. A 60–90 second loop at 96–128 kbps mono is plenty
  for ambience and keeps first load quick.
- **Quiet at source.** The component caps volume at 0.32, but a track that is
  hot to begin with still sounds loud relative to the page.
