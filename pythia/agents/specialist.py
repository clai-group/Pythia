"""
pythia/agents/specialist.py

The Specialist Agent classifies each note as 0 or 1.
Temperature should be 0.0 for deterministic classification.
"""
from pythia.llm.base import BaseLLMBackend

SYSTEM_TEMPLATE = '''You are a clinical note classifier.
Your task: {task}

Standard Operating Procedure:
{sop}

Instructions:
- Read the clinical note carefully.
- Respond with ONLY the number 1 (positive) or 0 (negative).
- Do not explain your answer. Do not add any other text.

Clinical note:
{note}

Classification (1 or 0):'''


def _parse_prediction(raw: str) -> int:
    """
    Extract 0 or 1 from the model's raw text output.
    Tries exact match first, then looks for '1' or '0' in the string.
    Defaults to 0 if unparseable (conservative — avoids false positives).
    """
    cleaned = raw.strip()
    if cleaned in ('0', '1'):
        return int(cleaned)
    if '1' in cleaned:
        return 1
    if '0' in cleaned:
        return 0
    return 0   # default to negative if we cannot parse


class SpecialistAgent:
    def __init__(self, backend: BaseLLMBackend):
        self.backend = backend

    def classify_note(
        self,
        prompt: str,
        sop:    str,
        note:   str,
    ) -> tuple[int, str]:
        """
        Classify a single note.
        Returns (prediction: int, raw_output: str).
        raw_output is kept so diagnostics.py can compute output entropy.
        """
        full_prompt = SYSTEM_TEMPLATE.format(
            task = prompt,
            sop  = sop,
            note = note,
        )
        raw = self.backend.invoke(full_prompt)
        return _parse_prediction(raw), raw

    def classify_all(
        self,
        prompt:  str,
        sop:     str,
        records: list[dict],
    ) -> tuple[list[int], list[str]]:
        """
        Classify all records. Uses batch_invoke() if the backend supports it
        (i.e. VLLMBackend), otherwise falls back to sequential invoke() calls.
        """
        full_prompts = []
        for record in records:
            full_prompts.append(
                SYSTEM_TEMPLATE.format(
                    task = prompt,
                    sop  = sop,
                    note = record['visit_text'],
                )
            )
        # full_prompts = [
        #     SYSTEM_TEMPLATE.format(
        #         task = prompt,
        #         sop  = sop,
        #         note = record['visit_text'],
        #     )
        #     for record in records
        # ]
        # Use parallel batch if available
        if hasattr(self.backend, 'batch_invoke'):
            raw_outputs = self.backend.batch_invoke(full_prompts)
        else:
            raw_outputs = [self.backend.invoke(p) for p in full_prompts]
        predictions = [_parse_prediction(r) for r in raw_outputs]
        return predictions, raw_outputs
