#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
The old ms-swift generative baseline is disabled.

This project now follows the MemReranker/Qwen3-Reranker scoring path:
  score = softmax([logit_no, logit_yes])[yes]

That requires access to model logits at the final assistant token. The previous
swift script used generated text parsing, which is not equivalent to BCE
soft-label distillation and can collapse MSE/NDCG.

Use scripts/eval_baseline.sh instead. If you implement a logits-capable
ms-swift scorer later, wire it to the same yes/no logits score definition.
EOF

exit 2
