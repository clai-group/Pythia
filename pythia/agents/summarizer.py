
"""
pythia/agents/summarizer.py

The Summarizer Agent aggregates error critiques into a new prompt.
It is given the original prompt, the SOP, all critiques, and
optionally a failed prompt to avoid repeating.
"""
from pythia.llm.base import BaseLLMBackend

SUMMARIZER_TEMPLATE = '''You are a prompt engineer improving a clinical
text classifier.

Standard Operating Procedure (do not change this):
{sop}

Current prompt:
{current_prompt}

Errors (what went wrong with current prompt):
{critiques}

{failed_section}

Task: Write an improved version of the current prompt that addresses
the errors described above by focusing on key indicators of the condition's presence.
Rules:
  - Do not mention sensitivity, specificity, or classification metrics.
  - Do not mention critiques directly, incorporate their lessons.
  - Do not change the task from the SOP.
  - Emphasize positive clinical signs and patterns that indicate the condition.
  - Avoid framing the prompt around absences or exclusions unless necessary.
  - Output ONLY improved prompt text, nothing else.

Improved prompt:'''

FAILED_SECTION_TEMPLATE = '''Previously tried prompt (do NOT repeat this
approach, it made performance worse):
{failed_prompt}
'''


class SummarizerAgent:
    def __init__(self, backend: BaseLLMBackend):
        self.backend = backend

    def synthesise(
        self,
        sop:            str,
        current_prompt: str,
        critiques:      list[str],
        failed_prompt:  str | None = None,
    ) -> str:
        """
        Returns the new prompt string.
        """
        critiques_text = '\n\n'.join(
            f'Error {i+1}: {c}'
            for i, c in enumerate(critiques)
        )
        failed_section = (
            FAILED_SECTION_TEMPLATE.format(failed_prompt=failed_prompt)
            if failed_prompt else ''
        )
        full_prompt = SUMMARIZER_TEMPLATE.format(
            sop            = sop,
            current_prompt = current_prompt,
            critiques      = critiques_text,
            failed_section = failed_section,
        )
        return self.backend.invoke(full_prompt).strip()
