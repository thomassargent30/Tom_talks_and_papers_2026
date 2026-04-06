# QuantEcon Lecture Generator

This directory is set up to use a reusable VS Code Copilot prompt that converts a research paper (PDF or TeX) into a QuantEcon-style lecture written in MyST markdown.

## How to use it

1. **Place your paper in this directory** — it should be a `.pdf` or `.tex` file, e.g. `hansen_2012.pdf`.

2. **Make sure the two standard support files are also present here:**
   - `likelihood_ratio_process.md` — style reference (a complete example QuantEcon lecture)
   - `quant-econ.bib` — the master QuantEcon bibliography

3. **Open this directory as the workspace in VS Code.**

4. **Open the Copilot chat panel** (`Cmd+Shift+I` on macOS).

5. **Type `/` in the chat input** — a menu will appear. Select **"QuantEcon Lecture from Paper"**.

6. **Supply the paper filename as the argument** when prompted, e.g. `hansen_2012.pdf`.

## What Copilot will produce

Copilot will automatically:

- Read the full text of the paper
- Read `likelihood_ratio_process.md` to match QuantEcon style, structure, and mathematical conventions
- Read `quant-econ.bib` to reuse existing bibliography keys
- Write a new MyST markdown file (e.g. `hansen_2012.md`) containing:
  - Correct YAML front-matter for Jupytext/Jupyter Book
  - An Overview section motivating the paper's ideas
  - Sections covering the paper's key concepts and results
  - Runnable Python code cells illustrating every major concept
  - At least 3 exercises with full solutions at the end
- Write a `new-references.bib` file containing any bibliography entries from the paper that are **not** already in `quant-econ.bib` (if all references are already present, this file is omitted)
- Print a brief summary of what was produced

## MyST formatting rules (for manual edits)

If you edit the generated `.md` file by hand, keep these rules in mind:

- Every display math block must have a **blank line before `$$` and a blank line after `$$`**:

  ```
  Some text ends here.

  $$
  L_t = \prod_{i=1}^{t} \ell(w_i)
  $$

  Text continues here.
  ```

- Cite references with `{cite}` using the BibTeX key, e.g. `{cite}Hansen_2012`.
- Cross-reference other QuantEcon lectures with `{doc}`, e.g. `{doc}likelihood_ratio_process`.
- Exercises use the `{exercise}` / `{solution-start}` / `{solution-end}` MyST directives.

## Where the prompt lives

The prompt file is stored at:

```
~/Library/Application Support/Code/User/prompts/quantecon-lecture.prompt.md
```

Because it is in your VS Code user profile folder, it is available in **every workspace** you open and will travel with VS Code settings sync.
