# Multi-model label review

This is how the project checks labels with several LLMs before a person looks at
them. It was first used on the matched set; see
[label-review-v1.md](../reports/label-review-v1.md) for that run.

**Purpose:** find wrong or unclear labels cheaply, so a person spends time only on
the pairs worth reading. It's a filter, not validation. Never describe the result
as human-reviewed.

## Method

1. **Fix the method first.** Commit the reviewers, prompt rules, flagging rule and
   retry rule before any reviewer label exists.
2. **Use models from different families.** Exclude the model that wrote the labels.
3. **Keep reviewers blind.** Show only the label definitions, opaque IDs and the
   statements, in shuffled batches. Don't show author labels, cells, families, the
   hypothesis, or anything from the repo.
4. **Isolate each run.** Start from an empty directory. If a run uses a tool, void
   that model's review.
5. **Allow one retry,** only for malformed or incomplete output. Never repair a
   model's output by hand, even an obvious typo. Keep the first output.
6. **Flag broadly.** Flag a pair if any reviewer disagrees with the author on the
   target decision or answers ambiguous. One dissent is enough; don't use a
   majority vote.
7. **Leave decisions to the owner.** The owner settles every flagged pair. Neither
   the author nor a model does.
8. **Rerun once and report it.** Rerun the frozen test once with no other changes,
   and report it beside the original whatever it shows.
9. **Retire the reviewers for that set.** Models that helped label a set can't
   serve as LLM-judge baselines on it.

## What the first run taught us

- **Check reasoning limits first.** At default effort, one model spent its whole
  output limit on reasoning and returned nothing, so all reviewers were set to low
  effort before any labels existed.
- **Reviewers aren't stable.** One rerun changed 5 of 36 labels, including two that
  decided whether a pair was flagged. A single run understates the disagreement.
  Next time, run each batch at least twice and flag on any run's dissent.
- **Agreement can overstate quality.** Reviewers can share blind spots, and they
  only judge the pairs they're given. They can't say whether the set itself is a
  fair test.
- **Ruling blind would be stronger.** On the first run the owner saw the author's
  label and the dissent. Next time, show the pair alone first.
- **Validate outputs directly.** A background job's exit code can come from a pipe.
  Confirm the label count, valid labels and absence of tool events from the raw
  files.
