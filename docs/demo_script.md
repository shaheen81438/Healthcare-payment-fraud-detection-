# 3-Minute Demo Script

*Setup beforehand:* `make compose-up && make api & make dashboard` (or the compose serve stack);
browser tabs open to `:8501` (dashboard Home), `:8000/docs`, and MLflow `:5001`.

---

**[0:00–0:25] The problem + the data story.**
"CMS estimated about $29 billion of improper Medicare fee-for-service payments last year.
Payment-integrity teams attack that with a fixed number of investigators — so the problem isn't
just *detecting* bad claims, it's *ranking* them under capacity. I built this platform end to end:
real CMS provider data calibrates a synthetic claim population, and I inject twelve documented
fraud-waste-abuse patterns so every metric is honest about what it measures — detection of known
patterns, not real fraud discovery."

**[0:25–1:00] Dashboard — portfolio to queue.** *(Home page)*
"96,000 scored claims; 1,061 flagged at the operating point; about $228k estimated payment at
risk. The threshold isn't arbitrary — it's the max of an expected-net-value optimum, the
investigator-capacity cutoff, and a precision floor." *(Review Queue)* "Investigators work this
capacity-ranked queue — every claim arrives with plain-language reason codes, and their
approve/dismiss decisions get captured — that's the future label stream."

**[1:00–1:45] One claim, explained.** *(Claim Detail on a flagged claim)*
"For any claim: risk score, estimated dollars at risk, the deterministic rules it tripped, and a
SHAP breakdown — this one is high because it's an exact duplicate of an earlier paid claim at a
provider billing 2.5× its peer group. Same-window similar claims are right here, so the
investigator sees the original too. And in research mode I can reveal the injected ground truth —
this transparency is deliberate; the labels are synthetic and the docs say so everywhere."

**[1:45–2:20] The economics.** *(Threshold Simulator)*
"Here's the part I care about most: sliders for capacity, review cost, recovery rate,
false-positive cost. Reviewing the top 1% yields about $49k net on this window at 99% precision;
push capacity to 10% and marginal review value goes negative — the curve shows exactly where to
stop. Recall at 5% capacity is 84%."

**[2:20–2:50] MLOps.** *(MLflow, then Monitoring page)*
"Every run is tracked in MLflow — params, metrics, artifacts, git commit — and the production
model is served by registry alias with a local fallback. Evidently monitors input drift,
score-distribution PSI, provider-mix shifts; here's a simulated pricing-drift run escalating to
critical. Investigator decisions feed a delayed realized-precision report."

**[2:50–3:00] Close.**
"FastAPI service with validated schemas and SHAP explanations, one-command pipeline, CI with a
model-quality regression gate, Docker images, Azure Container Apps deployment scripts. The whole
build is honest about its one boundary — synthetic labels — and engineered like it isn't a demo."
