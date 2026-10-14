# Limitations

This file is written to be quoted in the limitations section of a paper. Everything here is a
property of the design or the dataset, not a temporary bug.

## 1. Dataset limitations

1. **Age and provenance.** CICIDS2017 was captured in July 2017 in a purpose-built testbed with
   scripted attacks. Attack tooling, traffic mixes and protocol usage have moved on; a detector
   trained on it does not transfer to a modern network without re-validation.
2. **No domains, malware, vulnerabilities or ATT&CK mappings.** The `MachineLearningCSV` release has
   no DNS names, file hashes, CVE identifiers or technique labels. The knowledge graph therefore
   contains IP, host, service, protocol, event and (optionally) technique entities only. Any
   domain/malware/vulnerability node would be fabricated, so none is created.
3. **Label noise and known flaws.** Public analyses document mislabelled flows, duplicate rows and a
   duplicated `Fwd Header Length` feature in the released CSVs. The pipeline removes exact
   duplicates and drops the duplicate column but cannot repair incorrect labels. Reported metrics
   therefore contain an unknown amount of label error.
4. **Extreme class imbalance and per-day attack skew.** Attack families are unevenly distributed
   across days (for example most DDoS and PortScan traffic is on Friday). A day-based split can put
   an entire family on one side, so per-class results must be read alongside the recorded
   train/test label distributions.
5. **Prototype subset.** By default only a bounded number of rows per file are read. Metrics from a
   subset are **not** comparable to papers using all 2.83 million flows. The subset size is recorded
   in `dataset.sampling`.
6. **Synthetic fallback.** If no CSV is present, a synthetic generator substitutes for the dataset.
   Results then characterise the generator and must never be presented as detection performance.
   The flag `dataset.is_synthetic` and the generated report text make this explicit.

## 2. Knowledge graph limitations

7. **Relationships are co-occurrence, not causation.** `COMMUNICATES_WITH` means "flows were
   observed between these addresses in this window". It does not imply intent, coordination or
   control. No edge in the graph is asserted without a supporting record.
8. **Confidence values are heuristics, not probabilities.** `aggregated = 1 − exp(−n/K)` and the
   Laplace-smoothed intel ratio are monotone, bounded, documented functions of evidence counts.
   They were **not** calibrated against ground truth, so they rank evidence plausibility rather than
   expressing a true probability.
9. **Threat intelligence is hindsight.** The intel index is derived from **training labels**, i.e. a
   perfect retrospective feed over the training window. A live deployment would depend on an
   external feed with unknown coverage and latency. Features and explanations built on it are marked
   `uses_training_labels: true` and disclosed per alert.
10. **Internal-host modelling is simplified.** Every internal IP is mapped 1:1 to a host entity.
    There is no DHCP/NAT resolution, so "host" means "address in this capture", not "machine".
11. **Event nodes are capped.** Above `graph.max_event_nodes` only the first N events become nodes;
    the truncation is logged in the graph build log.
12. **Community detection is non-deterministic in principle.** Louvain is seeded and its method
    string is recorded, but community identity is not stable across different graph scopes.

## 3. Graph-aware detection limitations

13. **Transductive vs inductive.** In the default `full` scope the structural graph is built over
    all evaluated events — including test-period flows. Labels never enter, but test topology does.
    This flatters the graph-aware arm relative to a live deployment. The `leakage_sensitivity` block
    quantifies the gap by re-running with `train_only`; **report both**.
14. **No claim of superiority by default.** The comparison is a single dataset, a single split
    family and a handful of seeds. The rule-based claim only fires when the paired-bootstrap CI
    excludes zero, and even then it is a statement about *this* test split. It is not evidence of
    general graph superiority.
15. **Feature-count confound.** The graph-aware arm has more features. Any difference could come
    from extra capacity rather than relational information. Testing that would require an
    ablation with matched dimensionality (for example graph features only, or random noise features
    of equal count), which this prototype does not include.
16. **No hyperparameter or threshold tuning for either arm.** Both use defaults; the decision
    threshold is fixed at 0.5 for both. Results are therefore not the best each arm could do, but
    the comparison is at least symmetric.
17. **Single-algorithm design.** Both arms are RandomForests. Conclusions say nothing about how
    gradient boosting, neural detectors or unsupervised methods would behave.
18. **No drift, adversarial or robustness analysis.** There is no evaluation of evasion, poisoning,
    concept drift or performance decay over time.

## 4. Explanation limitations

19. **Attributions may be an approximation.** If SHAP is installed, per-alert values come from
    `shap.TreeExplainer`. Otherwise the explainer falls back to *global tree importance × local
    standardised deviation*, which is **not** Shapley-additive. The method name and a note are
    attached to every alert, and the limitations list repeats it. Never present the fallback as SHAP.
20. **Model explanations are not causal explanations.** They describe which inputs moved the score.
    They do not explain why traffic occurred, and they cannot distinguish a benign service that
    looks like an attack from an actual attack.
21. **Graph paths are proximity context.** A short path to an IP with historical malicious labels
    does not indicate the analysed event is malicious; benign hosts legitimately talk to previously
    attacked hosts. The path text says so, and the code never converts path presence into a verdict.
22. **No proof of intent.** Every explanation carries a disclaimer, because a statistical detection
    is not proof and should not be treated as sufficient grounds for enforcement.
23. **Explanations inherit dataset gaps.** Without domain/malware entities, an analyst sees
    address-level context only. The absence of a path or relationship is a coverage statement, not
    a clean bill of health.
24. **Alert selection is score-based.** Only the highest-scoring positive decisions are explained
    (default 25). Missed detections get no explanation, so the alert set is not a representative
    sample of model behaviour.

## 5. Measurement limitations

25. **Single machine.** All resource numbers come from the machine recorded in the `environment`
    block. They are not portable to other hardware, and background load on that machine is included
    in wall-clock times.
26. **CPU percent semantics.** Values are relative to one logical core and can exceed 100% when
    several cores are used. CPU *time* (`cpu_seconds`) is the more stable comparison.
27. **Peak RAM is a lower bound on some platforms.** Where the OS does not expose peak working set,
    the value comes from periodic sampling and can miss short allocation spikes.
28. **`tracemalloc` is not total memory.** It tracks Python-level allocations only, after tracing
    started, and misses buffers allocated in C extensions (NumPy/pandas/BLAS). It is reported as its
    own field and must not be described as process RAM.
29. **Few repeated runs.** The default is three. Mean ± std over three runs gives a rough sense of
    variability, not a tight interval. Seeds vary across runs, so variance includes model-fitting
    randomness, not just measurement noise.
30. **Warm-cache effects.** Repeated training runs reuse warm OS file caches and CPU frequency
    scaling states. First-run and later-run timings are not interchangeable.
31. **Explanation cost is amortised.** Explanation timing covers the configured number of alerts,
    not every event, and the SHAP explainer is constructed once outside the measured block.

## 6. Methodological limitations

32. **The leakage audit is a guard, not a proof.** It verifies split-level properties (no shared
    records, no label columns in features, chronological ordering, train-only fitting). It cannot
    detect subtler distributional leakage, and it cannot certify that the split reflects real
    deployment conditions.
33. **One dataset, one split family.** No cross-dataset or cross-year validation is performed.
34. **Statistically paired comparisons are still exploratory.** The bootstrap and McNemar tests are
    computed on the same test set used to select the reported configuration, which is not an
    independent validation set.
35. **No cost model.** The overhead table reports time, CPU and RAM. It does not translate into money
    or analyst time, and no alert-capacity or triage-workload analysis is included.

## 7. Honesty rule

If a component did not execute successfully, the artifacts mark it as `not_executed` or list it in
`failed_stages`, and its metrics are `null`. **No value in this repository is estimated and presented
as measured.** If you need a number that is missing, run the pipeline — do not fill it in by hand.
