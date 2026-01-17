### Merged CSV for IRT training ###
`FINAL_MERGED_FOR_IRT.csv` (merged dataset used for IRT training):
https://drive.google.com/file/d/15E4I5ELFg5poe85EACOxK5beK0Z2HC3b/view?usp=sharing

### Full data drive (all CSVs) ###
Includes the full merged CSV plus per-model CSVs:
Folder: https://drive.google.com/drive/folders/16LS-EUjxg4dYYtA3nLt-KzltwC8-rorp?usp=sharing


### Anchor Generation Prompt (Qwen3-32B): ###
system_prompt = (
    "You are a translation quality auditor. Compare the semantic meaning of the provided sentences. "
    "Ignore minor punctuation or tone differences. Focus on key intent and nouns."
)

user_prompt = f"""
Do the following four sentences have the EXACT same semantic meaning?
Definition: TRANSLATION DOES NOT ALTER THE SEMANTIC MEANING IN ANY WAY WHATSOEVER.
To any person native in both languages, they would mean the exact same thing.

1. English: "{row['en']}"
2. Chinese: "{row['zh']}"
3. Arabic: "{row['ar']}"
4. Bengali: "{row['bn']}"

Respond with ONLY "YES" or "NO". Do not provide explanations.
"""
