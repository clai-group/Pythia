"""
pythia/agents/improver.py

The Improver Agent performs counterfactual error analysis.
For each misclassified note it produces a critique: a short
natural-language explanation of what the prompt missed.
"""
from pythia.llm.base import BaseLLMBackend

SENSITIVITY_TEMPLATE = '''You are reviewing a classification error.
The prompt below was used to classify clinical notes, but it produced a
FALSE NEGATIVE: the note below describes a patient who HAS the condition,
but the prompt caused the classifier to say 'no'.

Current prompt: {prompt}

Note that was incorrectly classified as NEGATIVE:
{note}

Task: In 2-3 sentences, explain what signals in the note indicate the
condition is present, and what the prompt is missing that caused it to
be overlooked. Focus on specific language, phrases, or clinical patterns.

Critique:'''

SPECIFICITY_TEMPLATE = '''You are reviewing a classification error.
The prompt below was used to classify clinical notes, but it produced a
FALSE POSITIVE: the note below describes a patient who does NOT have
the condition, but the prompt caused the classifier to say 'yes'.

Current prompt: {prompt}

Note that was incorrectly classified as POSITIVE:
{note}

Task: In 2-3 sentences, explain what features of the note suggest the
condition is absent, and what in the prompt caused it to incorrectly
flag this note. Focus on what negative indicators were missed.

Critique:'''

class ImproverAgent:
    def __init__(self, backend: BaseLLMBackend):
        self.backend = backend

    def critique_errors(
        self,
        prompt: str,
        error_notes: list[str],
        target: str, # 'sensitivity' or 'specificity'
    ) -> list[str]:
        """
        For each error note, generate a critique.
        Returns list of critique strings, one per note.
        """
        template = (SENSITIVITY_TEMPLATE if target == 'sensitivity'
                    else SPECIFICITY_TEMPLATE)
        critiques = []
        for note in error_notes:
            full_prompt = template.format(prompt=prompt, note=note)
            critique    = self.backend.invoke(full_prompt)
            if critique.strip():          # filter empty responses
                critiques.append(critique.strip())
        return critiques
