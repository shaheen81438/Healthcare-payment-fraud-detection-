# 3-minute demo script

Setup beforehand: `make compose-up && make api & make dashboard` (or the compose serve stack),
with browser tabs open to `:8501` (dashboard Home), `:8000/docs`, and MLflow `:5001`.

**[0:00-0:25] The problem and the data.**
"CMS estimated about $29 billion of improper Medicare fee-for-service payments last year.
Payment-integrity teams handle that with a fixed number of investigators, so the problem is
*ranking* bad claims under capacity as well as *detecting* them. I built this platform end to end.
Real CMS provider data calibrates a synthetic claim population, and I inject twelve documented
fraud-waste-abuse patterns. So every metric measures detection of known patterns, not real fraud
discovery."

**[0:25-1:00] Dashboard, from portfolio to queue.** *(Home page)*
"96,000 scored claims, 1,061 flagged at the operating point, and about $228k estimated payment at
risk. The threshold is the max of an expected-net-value optimum, the investigator-capacity cutoff,
and a precision floor." *(Review Queue)* "Investigators work this capacity-ranked queue. Every
claim comes with plain-language reason codes, and their approve/dismiss decisions are captured.
Those decisions become the future label stream."

**[1:00-1:45] One claim, explained.** *(Claim Detail on a flagged claim)*
"For any claim you get the risk score, estimated dollars at risk, the deterministic rules it
tripped, and a SHAP breakdown. This one is high because it's an exact duplicate of an earlier paid
claim at a provider billing 2.5× its peer group. Similar claims from the same window are right
here, so the investigator sees the original too. In research mode I can reveal the injected ground
truth. I show it on purpose: the labels are synthetic and the docs say so everywhere."

**[1:45-2:20] The economics.** *(Threshold Simulator)*
"This is the part I care about most. There are sliders for capacity, review cost, recovery rate,
and false-positive cost. Reviewing the top 1% yields about $49k net on this window at 99%
precision. Push capacity to 10% and marginal review value goes negative; the curve shows where to
stop. Recall at 5% capacity is 84%."

**[2:20-2:50] MLOps.** *(MLflow, then Monitoring page)*
"Every run is tracked in MLflow (params, metrics, artifacts, git commit), and the production model
is served by registry alias with a local fallback. Evidently monitors input drift,
score-distribution PSI, and provider-mix shifts. Here's a simulated pricing-drift run escalating to
critical. Investigator decisions feed a delayed realized-precision report."

**[2:50-3:00] Close.**
"There's a FastAPI service with validated schemas and SHAP explanations, a one-command pipeline,
CI with a model-quality regression gate, Docker images, and Azure Container Apps deployment
scripts. The main limit is that the labels are synthetic, and the docs state that everywhere
results appear. Apart from that, I built it the way I'd build a production system."
