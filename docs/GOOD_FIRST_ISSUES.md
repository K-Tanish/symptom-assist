# Good first issues — plain-language descriptions

Short explanations you can reuse for GitHub issues or onboarding. Each item helps SymptomAssist stay reliable, clear, and easy for new contributors to work with.

Under each issue, **Suggested labels** are GitHub label ideas—create them in your repo if they don’t exist yet.

---

## 1. Session storage is declared twice

**Suggested labels:** `good first issue`, `cleanup`, `backend`

**What’s going on**

The app keeps chat sessions in memory (symptom history and when the session was last used). In the main server file, that storage variable is created twice in a row—the first line is immediately replaced by the second. The old comment also doesn’t match how sessions actually work now.

**Why it matters**

Dead code and mismatched comments confuse people reading the code and make it harder to reason about sessions or fix bugs later.

**What a contributor can do**

Remove the redundant first declaration and keep one clear comment that describes the real shape of the data (session id → symptoms + last active time, plus the expiry rule).

---

## 2. Symptom matching in the knowledge graph runs cleanup in the wrong place

**Suggested labels:** `bug`, `knowledge-graph`, `backend`

**What’s going on**

When turning the user’s symptoms into nodes on the graph, the code sometimes deduplicates its internal list in the middle of a loop—and some paths through the loop skip that step entirely. That makes behavior depend on which branch ran last. Using a plain `set` here also throws away a stable order, which can subtly affect how traversal proceeds.

**Why it matters**

SymptomAssist’s whole pitch is grounded, explainable reasoning from the graph. Inconsistent or order-sensitive matching weakens trust in that layer—even if users don’t see a mismatch directly.

**What a contributor can do**

Finish collecting all matched symptoms first, then deduplicate once using logic that’s easy to follow and deterministic (the code already tracks “seen” symptoms in a set—that’s a good hint).

---

## 3. Dependencies are listed twice in requirements

**Suggested labels:** `good first issue`, `chore`, `dependencies`

**What’s going on**

The main requirements file repeats the same packages back-to-back (for example Groq, FastAPI, and uvicorn appear twice).

**Why it matters**

New contributors may wonder which block is “correct,” installs look sloppy, and it’s harder to keep versions consistent when everything must be listed twice by mistake.

**What a contributor can do**

Merge into a single clean list. Optionally add minimum versions for important libraries so everyone gets a similar environment—especially helpful for RAG and embeddings.

---

## 4. Some crashes write a log file in the wrong place

**Suggested labels:** `bug`, `developer-experience`, `logging`, `backend`

**What’s going on**

If the chat endpoint hits an unexpected error, the server writes a file named `error_log.txt` in whatever folder the process was started from. Elsewhere, the app already uses a dedicated `logs` folder with rotating log files.

**Why it matters**

People running the app from different directories get mystery files in random places. It’s also inconsistent with the rest of logging, which makes debugging and support harder.

**What a contributor can do**

Route that traceback through the existing logging setup (or write under the same `logs` directory used elsewhere) instead of the current working directory.

---

## 5. Tests exist but aren’t run automatically on GitHub

**Suggested labels:** `enhancement`, `ci`, `testing`, `help wanted`

**What’s going on**

The repo already has automated tests (for example graph matching and UI safety checks), but there’s no GitHub Actions workflow that runs them when someone opens a pull request.

**Why it matters**

Without CI, contributors and reviewers can’t quickly see if a change broke graph logic or frontend assumptions. For a health-adjacent tool, small regressions are worth catching early.

**What a contributor can do**

Add a workflow that installs dependencies and runs `pytest` on pushes and pull requests. If some installs are heavy (for example optional NLP models), the issue description can note documenting skips or a lighter “CI requirements” path—whatever keeps the first version mergeable and maintainable.

---

## 6. (Major) “Take this to your appointment” summary

**Suggested labels:** `enhancement`, `priority`, `feature`, `frontend`, `backend`, `documentation`, `help wanted`

**What’s going on**

The app is good at walking someone through symptoms in chat and showing technical detail in the side panel, but there is no simple outcome most people actually need next: **a short, sober summary they can hand to a clinician or caregiver**—symptoms in order, how long things have bothered them if known, what the tool’s graph surfaced as possibilities (clearly framed as *not* a diagnosis), any red-flag wording, and which educational sources were used. Today that means scrolling, copying piecemeal, or re-explaining everything from memory.

**Why it matters**

SymptomAssist is explicit that it is **not** a substitute for professional care. The honest next step for users is often “see a doctor and explain clearly.” A visit-ready summary closes that gap instead of leaving people with a long chat log nobody in a ten-minute consult will read. It also reinforces what makes this project different from a generic bot: the graph-ranked possibilities and document titles are **visible in one place**, so the conversation feels grounded in something shareable, not just persuasive text.

**What a contributor can do (split across PRs if needed)**

1. **Backend** — Build a stable block of plain text (and optionally a small JSON structure) assembled from data you already return: symptom timeline, top graph matches with scores, red flags, RAG titles. Keep language cautious (“possible matches,” “not a diagnosis”) and reuse wording aligned with your disclaimer.

2. **Frontend** — Add a clear action: “Copy summary” and/or “Print-friendly view” that renders that block without chat chrome. No fancy PDF library required at first.

3. **Docs** — One paragraph in the README on what the summary is for—and what it must **not** be used for—so contributors don’t accidentally turn it into something that sounds like a medical record.

This is a deliberately practical milestone: it serves real users, strengthens the product’s positioning next to professional care, and mostly composes existing signals instead of inventing a new abstraction layer.
