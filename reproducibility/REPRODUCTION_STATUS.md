# Reproduction status

Material audit: 2026-10-08. These are review materials. The original dataset
identity and diagnostic summaries can be checked independently; a complete
reproduction of every manuscript result has not been established.

## Confirmed

- The 660 CSV files comprise six pseudonymous participants, 11 classes and 10
  repetitions in each participant-class cell. Their SHA-256 values are listed.
- Acquisition was nominally 50 Hz over two seconds; saved windows are
  resampled to 100 points per channel. Each CSV contains one independent
  recording, not an overlapping window from a continuous stream.
- The five-seed raw summary records mean Macro-F1 0.7931333 and accuracy
  0.8139394. Its standard deviations are 0.0098849 and 0.0096661. These values
  must not be replaced by different manuscript means without a matching
  original result source.
- The separate seed-810 offline LOSO gate audit records one explicit
  low-confidence Static fallback, 64 Static predictions and six non-Static
  windows predicted as Static among 660 windows. This is a diagnostic run,
  not an assertion of 660 physical-device trials.
- Raw robustness results are preserved without changing their clean baseline.
  The original independent AWGN clean mean Macro-F1 is 0.7585634; the separate
  drift rerun uses its own clean baseline. No result is shifted to a headline
  performance value by this package.

## Remaining version correspondence

- The archived primary performance checkpoint and exact historical inner
  train/validation file ordering have not been identified. Current-protocol
  manifests do not resolve that gap or guarantee the archived headline score.
- The offline legacy descriptor evaluates periodograms on a coordinate with
  `fs=250`. Physical acquisition was 50 Hz. The implementation is preserved
  as recorded; Python/MCU feature-coordinate equivalence remains unresolved.
- The included `BSL_Gesture_Demo` is the existing hardware demo with its model
  headers. Its confidence selector uses threshold 0.60 and probability
  averaging when both branches are low-confidence. It differs from the
  separate offline 0.5-threshold, margin/Static-fallback audit. This upload does
  not claim that the demo implements that audit rule.
- Full training dependencies have not been used to retrain all models here.
  The audit environment validates file identity, split isolation, descriptor
  dimensionality and raw-result arithmetic. It does not establish bitwise
  equivalence to a historical training environment.

See `verification.json` for performed checks. Treat a script, a diagnostic
output and a deployed binary as separate versioned inputs until their
correspondence has been verified.
