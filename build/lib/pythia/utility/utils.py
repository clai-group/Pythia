import re

def is_meaningful_paragraph(text):
    if not isinstance(text, str):
        return False

    # Must have at least 3 words
    word_count = len(text.split())
    if word_count < 3:
        return False

    # Must have sentence-like punctuation
    if not re.search(r'[,.!?]', text):
        return False

    # Must not be mostly digits
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.5:
        return False

    # Avoid high repetition 
    words = text.lower().split()
    unique_word_ratio = len(set(words)) / len(words) if words else 0
    if unique_word_ratio < 0.3:
        return False

    return True


def clean_prompt(content):
    colon_quote_match = re.search(r':\s*"', content)

    if colon_quote_match:
        # Find the position after the colon
        colon_pos = content.find(':', colon_quote_match.start())
        # Return everything after the colon and remove quotes
        result = content[colon_pos + 1:].replace('"', '').strip()
        # print(result)
        return result

    else:
        # print(content.replace('"', ''))
        return content.replace('"', '')


