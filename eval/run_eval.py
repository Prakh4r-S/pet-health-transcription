"""
Measures how well the extraction step turns transcripts into notes.

Run:  python eval/run_eval.py
      python eval/run_eval.py --provider gemini --model gemini-3.6-flash

Each case in eval/cases/ pairs a transcript with the extraction a vet
would consider correct. Writing those by hand is the tedious, unglamorous
part, and it is also the only thing that makes the numbers below mean
anything.

Three things are measured, and they answer different questions:

  Field decisions — did the model fill a field that should be filled,
  and leave null one that should be null? This is the coarse question and
  the one that catches the worst failure: inventing content for something
  the consultation never covered.

  Field content — when it did fill a field, how much of the expected
  content is there? Token F1 rather than exact match, because two vets
  would not write the same sentence and neither would two runs of a
  model. It is a blunt instrument and should be read as "roughly right"
  rather than as a quality score.

  Medications — precision and recall on drug names, and exact-match
  accuracy on dose and frequency. These are separated deliberately.
  Getting a drug name wrong is bad; getting a dose wrong is dangerous,
  and averaging the two into one number hides exactly the thing you most
  need to see.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract import ProposedNote, extract_note  # noqa: E402

CASES_DIR = Path(__file__).parent / "cases"
TEXT_FIELDS = ["presenting_complaint", "examination", "assessment", "plan", "follow_up"]


def tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    # Lowercase, strip punctuation, drop the filler words that inflate
    # any overlap score without carrying clinical meaning.
    stop = {"the", "a", "an", "and", "of", "to", "is", "was", "on", "in", "for", "with"}
    words = re.findall(r"[a-z0-9.]+", text.lower())
    return {w for w in words if w not in stop}


def token_f1(predicted: str | None, expected: str | None) -> float:
    p, e = tokens(predicted), tokens(expected)
    if not p and not e:
        return 1.0
    if not p or not e:
        return 0.0
    overlap = len(p & e)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(e)
    return 2 * precision * recall / (precision + recall)


def normalise_drug(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


@dataclass
class Totals:
    decisions_correct: int = 0
    decisions_total: int = 0
    # Fields the model filled that should have been null. Tracked apart
    # from other errors because this is the failure that puts invented
    # clinical text in front of a vet.
    hallucinated_fields: int = 0
    missed_fields: int = 0
    f1_scores: list[float] = field(default_factory=list)

    drugs_predicted: int = 0
    drugs_expected: int = 0
    drugs_matched: int = 0
    dose_correct: int = 0
    dose_comparable: int = 0
    # Doses stated in the transcript that the model got wrong. The single
    # most consequential error the pipeline can make.
    dose_wrong: list[str] = field(default_factory=list)


def score_case(predicted: ProposedNote, expected: dict, totals: Totals, name: str) -> dict:
    per_field = {}

    for f in TEXT_FIELDS:
        got = getattr(predicted, f)
        want = expected.get(f)

        got_filled = bool(got and got.strip())
        want_filled = bool(want and str(want).strip())

        totals.decisions_total += 1
        if got_filled == want_filled:
            totals.decisions_correct += 1
        elif got_filled and not want_filled:
            totals.hallucinated_fields += 1
        else:
            totals.missed_fields += 1

        if want_filled:
            f1 = token_f1(got, want)
            totals.f1_scores.append(f1)
            per_field[f] = round(f1, 2)

    expected_meds = {normalise_drug(m["drug"]): m for m in expected.get("medications", [])}
    predicted_meds = {normalise_drug(m.drug): m for m in predicted.medications}

    totals.drugs_expected += len(expected_meds)
    totals.drugs_predicted += len(predicted_meds)

    for key, want_med in expected_meds.items():
        got_med = predicted_meds.get(key)
        if not got_med:
            continue
        totals.drugs_matched += 1

        # Only score a dose when the transcript actually stated one.
        # Penalising a null against a dose that was never said would
        # reward guessing, which is the opposite of what this pipeline
        # should do.
        if want_med.get("dose"):
            totals.dose_comparable += 1
            if got_med.dose and got_med.dose.strip().lower() == want_med["dose"].strip().lower():
                totals.dose_correct += 1
            else:
                totals.dose_wrong.append(
                    f"{name}: {want_med['drug']} expected {want_med['dose']!r}, got {got_med.dose!r}"
                )

    return {
        "field_f1": per_field,
        "drugs_expected": sorted(expected_meds),
        "drugs_found": sorted(predicted_meds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=os.environ.get("LLM_PROVIDER", "gemini"))
    parser.add_argument("--model", default=os.environ.get("EXTRACTION_MODEL"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["EXTRACTION_MODEL"] = args.model

    cases = sorted(CASES_DIR.glob("*.json"))
    if not cases:
        print(f"No cases found in {CASES_DIR}")
        return

    totals = Totals()
    print(f"Running {len(cases)} cases through {args.provider}"
          f"{f' / {args.model}' if args.model else ''}\n")

    skipped: list[str] = []

    for path in cases:
        case = json.loads(path.read_text())

        if not case["transcript"].strip():
            # Cases 13-18 wait on a recording. Counting them as failures
            # would quietly depress every number in the report.
            skipped.append(f"{path.stem} (no transcript yet)")
            continue

        try:
            predicted = extract_note(case["transcript"])
        except Exception as exc:  # noqa: BLE001
            # One flaky API call should not throw away the whole run.
            skipped.append(f"{path.stem} ({exc})")
            print(f"  {path.stem}: SKIPPED — {exc}")
            continue

        detail = score_case(predicted, case["expected"], totals, path.stem)

        if args.verbose:
            print(f"  {path.stem}: {detail}")
            # An all-zero row means the model returned nothing usable,
            # which is a different failure from getting the content
            # wrong — worth seeing rather than averaging away.
            if detail["field_f1"] and all(v == 0.0 for v in detail["field_f1"].values()):
                print(f"    ^ empty or unparsed extraction. Raw: {predicted!r}")
        else:
            print(f"  {path.stem}: done")

    if skipped:
        print(f"\nSkipped {len(skipped)} case(s):")
        for line in skipped:
            print(f"  {line}")

    if totals.decisions_total == 0:
        print("\nNo cases scored.")
        return

    print("\n--- Field decisions ---")
    print(f"Correct filled/null decisions : {totals.decisions_correct}/{totals.decisions_total}"
          f" ({totals.decisions_correct / totals.decisions_total:.0%})")
    print(f"Invented (should be null)     : {totals.hallucinated_fields}")
    print(f"Missed (should be filled)     : {totals.missed_fields}")

    if totals.f1_scores:
        mean_f1 = sum(totals.f1_scores) / len(totals.f1_scores)
        print(f"\n--- Field content ---")
        print(f"Mean token F1 on filled fields: {mean_f1:.2f}")

    print("\n--- Medications ---")
    precision = totals.drugs_matched / totals.drugs_predicted if totals.drugs_predicted else 0.0
    recall = totals.drugs_matched / totals.drugs_expected if totals.drugs_expected else 0.0
    print(f"Drug name precision : {precision:.2f} ({totals.drugs_matched}/{totals.drugs_predicted})")
    print(f"Drug name recall    : {recall:.2f} ({totals.drugs_matched}/{totals.drugs_expected})")

    if totals.dose_comparable:
        print(f"Dose exact match    : {totals.dose_correct}/{totals.dose_comparable}"
              f" ({totals.dose_correct / totals.dose_comparable:.0%})")

    if totals.dose_wrong:
        print("\nDose errors — read every one of these:")
        for line in totals.dose_wrong:
            print(f"  {line}")


if __name__ == "__main__":
    main()
