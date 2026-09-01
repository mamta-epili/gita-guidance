---
name: brd-prd
description: Write or revise Business Requirements Documents (BRDs) and Product Requirements Documents (PRDs) in this project's house format. Use when the user asks for a BRD, PRD, requirements doc, product spec, or business case; when they want to add a section, requirement, or decision record to an existing one; or when they ask "what's the difference between a BRD and a PRD". Keeps new documents consistent with docs/BRD.md and docs/PRD.md.
---

# BRD / PRD authoring

House format for requirements documents in this project. Reference
implementations: [`docs/BRD.md`](../../../docs/BRD.md) and
[`docs/PRD.md`](../../../docs/PRD.md) — read them before writing a new one, and
match their structure.

## The split

| | BRD | PRD |
|---|---|---|
| Answers | **Why** build this, and what business outcome | **What** it does and **how** it's specified |
| Audience | Sponsors, stakeholders, future-you deciding whether to continue | Whoever builds it |
| Owns | Problem, objectives, success metrics, scope, constraints, risks, phases | Users, functional requirements, architecture, NFRs, acceptance criteria, milestones |
| Lifespan | Changes rarely; a change means the *business case* changed | Changes constantly |

The most common failure is a PRD that re-argues the business case and a BRD full
of implementation detail. When in doubt: **if a competitor's engineer could
implement it from the sentence, it's PRD. If it explains why anyone should care,
it's BRD.**

Cross-reference rather than duplicate. Scope lives in the BRD; the PRD links to
it. Assumptions live in the BRD §0; the PRD says "BRD §0 applies".

## Structure

**BRD:** metadata table → assumptions (§0) → executive summary → problem
statement → scope (in / out / explicitly rejected) → business objectives →
success criteria → constraints → risks → phases → open questions.

**PRD:** metadata table → users and jobs → product principles → decision records
→ functional requirements → architecture → non-functional requirements →
acceptance criteria → milestones → out of scope → open questions.

Both open with the same metadata table: document type, product, owner, status,
last updated, related docs.

## Rules that make these documents useful

**1. Assumptions get their own section, at the top, flagged.** Every inferred
fact goes in a table with what to change if it's wrong. This is the single
highest-value section, because it makes wrong assumptions cheap to correct
instead of silently load-bearing.

**2. Every requirement gets an ID.** `FR-2.3`, `BO-1`, `NFR-4`, `R-1`. Without
IDs you cannot reference them in tickets, tests, or arguments.

**3. Success criteria are numbers with two thresholds** — a target and a floor
that blocks shipping. "High quality" is not a criterion. "Recall@10 ≥ 0.90,
ship-blocked below 0.80" is.

**4. Scope has three lists, not two.** In scope, out of scope, and **explicitly
rejected with the reason**. The third list is what stops the same bad idea
returning every month.

**5. Risks carry mitigations, likelihood, and impact.** A risk without a
mitigation is a worry. Include the risks that are *certain* — a known limitation
recorded honestly is worth more than an optimistic omission.

**6. Write decision records for the counterintuitive calls.** Especially where
the obvious approach is wrong. State the decision, the disqualifying evidence in
a comparison table, the alternatives considered and why each was rejected, and
what value survives from the rejected path. See PRD §3 for the pattern.

**7. Open questions stay open.** Do not resolve them with a guess to make the
document look finished. Number them; they're a work queue.

**8. Tables over prose** for anything enumerable. Prose for reasoning.

## Honesty rules

These matter more than the formatting.

- **Never write a requirement premised on something that won't work.** If the
  user's stated plan has a technical flaw, put it in a decision record with the
  numbers, and tell them directly in conversation too. A PRD that quietly
  designs around a false premise is worse than no PRD.
- **Distinguish inferred from stated.** Anything you assumed goes in §0, not
  buried in a requirement.
- **Date the volatile recommendations.** Model names, benchmark rankings and
  pricing go stale in months. Write "current as of <month year>; re-check before
  <phase>" next to them.
- **Cite research.** If a recommendation came from a search, link the source in
  the table.

## Process

1. Confirm the assumptions you're about to make — ask if any is both uncertain
   and expensive to reverse
2. Research anything version-dependent before specifying it
3. BRD first, then PRD — the PRD's requirements should trace to BRD objectives
4. Markdown in `docs/`, git-versioned. Produce `.docx` only when asked, and keep
   the markdown as source of truth
5. Status stays `Draft vN.N` until the user says otherwise
