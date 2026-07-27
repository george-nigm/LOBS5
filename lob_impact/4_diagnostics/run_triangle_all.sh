#!/bin/bash
#SBATCH --job-name=lobimp_tri_all
#SBATCH --account=brics.u6gb
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact/4_diagnostics/logs/tri_all_%j.out
#
# Build control-triangle Word reports for a list of models and collect EVERY model's
# docx (freshly built + latest existing neural ones) into ONE stable download folder:
#   4_diagnostics/results/triangle_reports_all/
#
#   sbatch lob_impact/4_diagnostics/run_triangle_all.sh                      # 5 baselines
#   MODELS="GDN" sbatch lob_impact/4_diagnostics/run_triangle_all.sh        # add a model later
set -euo pipefail
IMPACT_DIR="${IMPACT_DIR:-/home/u6gb/georgenigm.u6gb/LOBS5/lob_impact}"
NS="${NS:-64}"
MODELS="${MODELS:-Historic Heuristic Propagator OW CST NMZI Hawkes QR S5_120M S5_4k Mamba3 Mamba3_4k GDN}"
COPY_EXISTING="${COPY_EXISTING:-Mamba3 Mamba3_4k S5_4k}"   # models whose latest docx is reused
RUN_TS="$(date +%Y%m%d-%H%M%S)"
RES="${IMPACT_DIR}/4_diagnostics/results"
ALL_DIR="${RES}/triangle_reports_all"
mkdir -p "${IMPACT_DIR}/4_diagnostics/logs" "$ALL_DIR"

set +u
source "${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-lobs5}"
set -u

cd "$IMPACT_DIR"
for M in $MODELS; do
  OUT="${RES}/triangle_${M}_${RUN_TS}"
  mkdir -p "$OUT"
  echo "[$(date)] building triangle for $M -> $OUT"
  # control_triangle_report.py needs the 'invisible' regime, which only exists for the models that
  # can condition on injected flow -- 4 of the 13 on EA. Under `set -e` the first model without it
  # aborted the WHOLE run (job 5789618 died on OW after Propagator had already succeeded, losing
  # every model queued behind it). Skip and warn instead, and report the skip list at the end.
  if ! python -u 4_diagnostics/control_triangle_report.py --model "$M" --n_samples "$NS" --out_dir "$OUT"; then
    echo "SKIP $M — control_triangle_report failed (usually: no invisible control for this model)" >&2
    SKIPPED="${SKIPPED:-} $M"
    continue
  fi
  python -u 4_diagnostics/make_triangle_docx.py --fig_dir "$OUT" --model "$M" \
      --out "$OUT/Control_Triangle_Report_${M}.docx"
  cp "$OUT/Control_Triangle_Report_${M}.docx" "$ALL_DIR/"
done
[ -z "${SKIPPED:-}" ] || echo "TRIANGLE_SKIPPED:${SKIPPED}"

# reuse the newest existing docx for models not rebuilt in this run
for M in $COPY_EXISTING; do
  latest="$(ls -d ${RES}/triangle_${M}_2*/ 2>/dev/null | sort | tail -1)"
  if [ -n "$latest" ] && [ -f "${latest}Control_Triangle_Report_${M}.docx" ]; then
    cp "${latest}Control_Triangle_Report_${M}.docx" "$ALL_DIR/"
    echo "copied existing ${M} docx from ${latest}"
  else
    echo "WARN: no existing docx for ${M}" >&2
  fi
done

echo "[$(date)] TRIANGLE_ALL_DONE -> $ALL_DIR"
ls -la "$ALL_DIR"
