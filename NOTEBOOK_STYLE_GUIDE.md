# Tutorial Notebook Style Guide

Internalized conventions from `00_introduction.ipynb`, `01_feature_attribution_methods.ipynb`, `02_structure_attribution_methods.ipynb`, and `03_counterfactual_explanations.ipynb`. Use this as the template when authoring a new tutorial notebook in this series.

---

## 1. Top-Level Skeleton

Every tutorial notebook (i.e. not the `00_introduction`) follows this fixed sequence:

1. **Title cell** (markdown)
   - Exactly: `# **Tutorial N** $\cdot$ <Topic Name>`
   - Immediately followed (same cell) by a blockquote abstract: `> **Abstract.** ...`
     - One dense paragraph. Names the methods covered, the dataset, what is demonstrated, and what the reader will see / compare.
   - Then `**💾 Dataset.**` paragraph: link to dataset, citation link to original paper, size, the property being predicted, performance caveats / known practical limits.
   - Then `**📦 Packages.**` followed by a bulleted list of packages used. Each bullet is a markdown link to the docs/repo plus a short sentence on why it is used.

2. **Single imports code cell**
   - Standard lib first (`os`, `random`, ...), then third-party (`torch`, `numpy`, `pandas`, `matplotlib`, `rdkit`, `lightning`, etc.), then project imports (`from xai_chem_review import load_dataset_<name>`).
   - Sometimes warning suppression at the end (`warnings.filterwarnings('ignore')`, `RDLogger.DisableLog('rdApp.*')`, `plt.style.use('default')`).

3. **Brief markdown** introducing the dataset loader function in one or two sentences.

4. **Dataset-loading code cell**
   - `data_frame: pd.DataFrame = load_dataset_<name>()`
   - `print(f'Loaded dataset with {data_frame.shape[0]} rows')`
   - `data_frame.head()` as last expression to render the table.

5. **Preprocessing code cell** (when applicable)
   - E.g. `data_frame['mol'] = data_frame['smiles'].apply(Chem.MolFromSmiles)` + drop-nans + print row count.

6. **Section `## **N.1** $\cdot$ 💡 <Concept Topic>`**
   - Conceptual motivation. No code. Explains *what kind* of explanation is being introduced and *why* it matters. Compares against previous tutorials when relevant.

7. **Sections `## **N.2** $\cdot$ 📚 *<Method Name>* (`pkg_name`)`, `## **N.3** ...`, etc.**
   - One section per method. Order follows: original paper citation (linked) → intuition → `**Implementation.**` paragraph naming the python package used.
   - Optional `<details>` blue info box (`📔`) with deeper theoretical background right after the intro.
   - One or more `### Subsection` H3s for the workflow phases. Typical phases (re-used across tutorials):
     - `### Molecular <Representation> Generation` (descriptors, fingerprints, graphs)
     - `### Model Training` (or a model-specific variant like `### Graph Neural Network Property Prediction`)
     - `### Generating <Method> Explanations`
   - Code cells interleaved with markdown explanation cells. Markdown either: (a) sets up what the next code cell does, or (b) interprets the output of the previous code cell (often starting with `**📊 <Plot Name>.**` or `**🔍 <Observation>.**`).
   - Exercises (gold boxes, see §3) interspersed where they fit naturally.

8. **Optional comparison section** `## **N.X** $\cdot$ 🔬 Comparison of Explanation Methods`
   - Only when the tutorial covers ≥3 methods. Side-by-side visualization on shared example molecules.
   - Ends with a `### Observations and Discussion` H3 inside the section.

9. **`## **N.Y** $\cdot$ 🔬 Discussion and Limitations`** (always the penultimate section)
   - 3–4 paragraphs. Each paragraph starts with a **bold inline header.** describing the concern (e.g. `**Chemical Validity of Perturbations.**`, `**Explanation Faithfulness.**`, `**Reproducibility.**`, `**Counterfactuals Explain the Model, Not the Chemistry.**`, `**Practical Implications.**`).
   - Tone: critical, balanced, names trade-offs. Refers back to the methods just demonstrated.

10. **`## **References**`** (final section)
    - Categorized: `**Datasets**`, `**Methods**`, `**Software**`. Full academic citations. Method entries end with `| [GitHub](https://...)` when a repo exists.

---

## 2. Section-Heading Convention

- Top tutorial header: `# **Tutorial N** $\cdot$ <Title>`
- Numbered sections: `## **N.M** $\cdot$ <emoji> <Title>` — the `$\cdot$` middle-dot LaTeX is mandatory.
- Emoji prefix encodes section type:
  - `💡` Concept / motivation
  - `📚` Method (named explanation technique)
  - `🔬` Discussion / comparison
- Sub-headings inside a numbered section use plain `### Title` (no number, no emoji prefix).
- Special unnumbered top-level sections: `## **References**`.

Inline-bold "labels" inside paragraphs follow the pattern `**Label.** content...` to break a long passage into named claims (`**Implementation.**`, `**Core Methodology.**`, `**Trade-offs.**`, `**Key Advantages.**`, `**Why Background Data?**`, etc.). The trailing period inside the bold is mandatory.

---

## 3. Reusable Markdown Blocks (copy-paste exactly)

### 🛠️ Exercise (gold box)

```html
<div style="background: #fff0c2; padding: 10px; border-style: solid; border-width: 1px; border-color: #dbad21; border-radius: 3px; color: black;">

**🛠️ Exercise N.M** $\cdot$ <prompt text>

</div>
```

- Numbered `N.M` continues across the whole tutorial (Exercise 1.1, 1.2, 1.3, 1.4 in tutorial 1).
- Prompts are intentionally additive — they ask the reader to tweak parameters or extend the analysis. No solutions provided.

### 📔 Additional Information (blue collapsible)

```html
<details style="border: 1.5px solid #536CCE; border-radius: 3px; padding: 10px; background-color:#EFF2FD; color: black; font-size: 0.9em;">
<summary style="cursor: pointer; font-weight: bold; color: #536CCE;">📔 <Topic Title></summary>

<deeper theoretical content with sub-bold labels and lists>

</details>
```

- Used for in-depth theory the average reader can skip (game-theoretic foundations, hyperparameter rationale, background-data choice rationale, etc.).
- Body uses the same `**Label.**` paragraph style.

### Post-plot interpretation cells

After a code cell that produces a figure, the next markdown cell starts with one of:

- `**📊 <Plot Name>.**` — explains how to read the axes/colors/encoding and points at the salient features the reader should see.
- `**🔍 <Observation>.**` — discusses what insights the reader should draw.
- `**🔬 <Theme>.**` — heavier interpretive discussion (used in MEGAN/comparison sections).

### 📝 Note

Inline within a markdown cell: `**📝 Note.** ...`. Used for caveats, library quirks (e.g. shap expects pytorch tensors not numpy arrays for torch models), or interpretation hints.

---

## 4. Code Style

### General

- Type hints **everywhere**. Including on local variables when it aids reading (`mol: Chem.Mol = Chem.MolFromSmiles(value)`, `values: list[float] = []`).
- `data_frame` (snake-case, full word) is the canonical name — never `df`.
- Constants UPPER_SNAKE_CASE at class level (e.g. `DESCRIPTOR_DETAIL_MAP`, `TARGET_SMILES`).
- `rich.pretty.pprint` for displaying complex objects (`pprint(shap_exp, max_length=10)`).
- Use `random.sample` for splits; convention: 20% test (`int(len(data_frame) * 0.2)`).

### Block-level comments inside cells

Code cells are subdivided with separator-style comments:

```python
# --- section name ---
```

Common labels: `# --- train-test split ---`, `# --- creating tensor dataset ---`, `# --- creating dataloaders ---`, `# --- setting up the explainer ---`, `# --- calculating SHAP values ---`, `# --- visualizing explanations ---`, `# --- example usage ---`, `# --- instantiating the model ---`, `# --- model training ---`.

Multiline plain `#` comments are used liberally above non-obvious lines to explain *why* (not just what). E.g.:

```python
# The `num_workers` argument specifies how many subprocesses to use for data loading. Together with a reasonable `prefetch_factor`, this
# ensures that the data is loaded efficiently during training - preventing performance bottlenecks.
```

### Class definitions

- Class-level docstring on the same line/triple-quote pattern as below.
- One-arg-per-line signatures when there is more than one non-trivial argument.
- Method docstrings always use sphinx-style: `:param name: ...` and `:return: ...` (sometimes `:returns:`).
- Convention: every "generator" class (e.g. `DescriptorGenerator`, `MorganFingerprintGenerator`) implements `__len__` and exposes a human-readable `descriptor_names` / `feature_names` property so explanations can be labelled meaningfully.

```python
class DescriptorGenerator:
    """
    A class to generate molecular descriptors for a given molecule.
    """

    DESCRIPTOR_DETAIL_MAP: dict[str, dict] = { ... }  # class-level constant config

    def get_descriptor_vector(self, value: str | Chem.Mol) -> list[float]:
        """
        Given the RDKit.Mol object `value`, this method returns ...

        :param value: Either a SMILES string or an RDKit.Mol object ...
        :return: A list of float values ...
        """
        ...

    # --- properties ---

    @property
    def descriptor_names(self) -> list[str]:
        """
        A list of all the human-readable descriptor names ...
        """
        ...

    def __len__(self) -> int:
        return len(self.DESCRIPTOR_DETAIL_MAP)
```

Always follow a class definition with a `# --- example usage ---` block in the same cell that constructs an instance and prints/`pprint`s a sample output. This previews behaviour before the class is used on the full dataset.

### PyTorch Lightning models

The recurring shape:

```python
class SimpleModel(pl.LightningModule):
    """
    <one-line summary>

    :param input_dim: ...
    :param output_dim: ...
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 64),         nn.BatchNorm1d(64),  nn.GELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x): ...
    def training_step(self, batch, batch_idx): ...
    def validation_step(self, batch, batch_idx): ...
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
```

After instantiating, print a "shape test":

```python
print('--- Testing the model ---')
x_example = torch.tensor(np.random.rand(10, len(descriptor_generator)), dtype=torch.float32)
y_example = model(x_example)
print(f'Example predictions for random input vectors (shape: {y_example.shape}):', y_example)
```

After training:

```python
trainer = pl.Trainer(max_epochs=25, accelerator='auto')
trainer.fit(model=model, train_dataloaders=loader_train, val_dataloaders=loader_test)
model.eval()
```

Always call `model.eval()` at end of training cell — this is consistent across all three tutorials.

---

## 5. Writing-Style Trends

- **Voice:** First-person plural "we" for actions (`we'll use`, `we can`, `we can create`). Second-person "you" reserved for the intro notebook addressing the reader directly.
- **Tense:** Present tense for describing what code does; future-ish ("we'll") when previewing the next step.
- **Em-dashes** use HTML entity `&mdash;` (not `—` or `--`).
- **Italics** for first introductions of technical terms (`*explanation*`, `*counterfactual*`, `*feature-attribution*`, `*structure-attribution*`).
- **Bold** for inline labels (`**Implementation.**`) and key terms within bulleted lists (`**Local vs. Global**:`, `**Trainability**:`).
- **Numbered lists** for itemising multi-point distinctions, e.g. "How It Affects Explanations", "Key Properties", "Core Methodology".
- **Method intros** follow a recurring three-beat opening: (1) attribution to the original authors with citation link, (2) one-paragraph plain-English explanation of the algorithm, (3) `**Implementation.**` paragraph naming the python package and its strengths.
- **Plot interpretation** never assumes the reader can decode a chart — always describe vertical axis, horizontal axis, color encoding, then point at the chemically-meaningful pattern that should be visible (e.g. "low values of ClogP are indicative of *increased* water solubility — these findings reflect the well-known anti-correlation between the octanol-water partition coefficient and the water solubility logS").
- **Discussion section paragraphs** are long but disciplined: bold-led claim, then 3–5 sentences elaborating, then a generalisation or actionable takeaway. They acknowledge weaknesses without undermining the methods.

---

## 6. Visual / Plotting Conventions

- `matplotlib` + `networkx` for graph visualization. RDKit `Draw` (incl. `Draw.MolsToGridImage`, `Draw.DrawMorganBit`, `Draw.DrawMorganBits`) for molecule rendering inside notebooks (via `IPython.display.display`).
- Side-by-side comparison plots use `plt.subplots(1, n_methods, figsize=(20, 5))`.
- Custom `draw_graph` / `draw_explanation` / `draw_explanation_megan` helpers in tutorial 2 — when reused in tutorial 3, the graph-processing functions are explicitly re-defined in the notebook (cells are self-contained; the tutorials *do not* assume cross-notebook state).
- Color conventions: `positive_color='lightgreen'` for positive attributions, red for negative or counter-class contributions, diverging colormaps for multi-channel (MEGAN) explanations.

---

## 7. Things That Tend to Appear in Every Tutorial

- A custom feature/representation generator class with `__len__`, a names property, and an `# --- example usage ---` demonstration.
- One model class (Lightning module), trained inline, with a `print('--- Testing the model ---')` shape-check before training.
- At least one `<details>📔` blue box of optional deeper background.
- At least 3–4 exercises in gold boxes spread through the tutorial.
- One `**📊 Plot.**` interpretation paragraph after every non-trivial figure.
- A `## **N.X** $\cdot$ 🔬 Discussion and Limitations` section with 3+ bold-headed paragraphs.
- A categorised `## **References**` section as the last cell.

---

## 8. Per-Tutorial Quick Reference

| # | File | Dataset | Methods covered | Distinctive subsections |
|---|------|---------|------------------|--------------------------|
| 0 | `00_introduction.ipynb` | — | — (meta) | Prerequisites, How These Tutorials Are Structured, Section Markers, Exercises, Additional Information, Getting Started |
| 1 | `01_feature_attribution_methods.ipynb` | AqSolDB (regression, logS) | SHAP (descriptors + MLP), LIME (Morgan fingerprints + MLP) | Molecular Descriptor Generation, Model Training, Generating Explanations, Morgan Fingerprint Generation |
| 2 | `02_structure_attribution_methods.ipynb` | Ames Mutagenicity (binary class.) | GNNExplainer, PGExplainer, Myerson values, MEGAN | Molecular Graph Processing (encode_atom / encode_bond, graph_from_smiles / data_from_graph), Graph Neural Network Property Prediction, Comparison of Explanation Methods |
| 3 | `03_counterfactual_explanations.ipynb` | AqSolDB (regression, logS) | MMACE (`exmol`), exhaustive 1-edit enumeration (`vgd_counterfactuals`) | Graph Neural Network Model (reuses tutorial-2 graph code), Installing and Setting Up ExMol, Generating Counterfactual Explanations, Comparing MMACE vs Exhaustive Enumeration |

---

## 9. Template for a New Tutorial

When creating, say, tutorial 4 on a new topic (e.g. uncertainty quantification), produce cells in this exact order:

1. Title + Abstract + `💾 Dataset` + `📦 Packages` (single markdown cell).
2. Imports (code).
3. Loader intro (markdown, 1–2 sentences).
4. `load_dataset_*` + `head()` (code).
5. `data_frame['mol'] = ...` (code).
6. `## **4.1** $\cdot$ 💡 <Concept Section>` (markdown, motivation, contrast to previous tutorials).
7. `## **4.2** $\cdot$ 📚 *<Method 1>* (`pkg`)` (markdown intro + `**Implementation.**` + optional `📔` blue box).
8. `### <Representation> / Model Training / Generating <Method 1>` subsections, alternating markdown/code, code cells liberally commented with `# --- ... ---` separators, custom classes with sphinx docstrings + `# --- example usage ---`.
9. Plot → `**📊 ...**` interpretation pattern.
10. Gold-box exercises (≥3 total per tutorial).
11. `## **4.3** $\cdot$ 📚 *<Method 2>* ...` if applicable.
12. `## **4.4** $\cdot$ 🔬 Comparison ...` if ≥3 methods.
13. `## **4.5** $\cdot$ 🔬 Discussion and Limitations` (3–4 bold-headed paragraphs).
14. `## **References**` categorised by Datasets / Methods / Software.

Match the present tense, "we"-voice, `&mdash;` dashes, italicised first-mentions, and the `**Label.**` paragraph style throughout.
