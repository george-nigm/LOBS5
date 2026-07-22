#!/usr/bin/env bash
# Сборка методического PDF (маркет-импакт + аудит β).
# Лёгкая CPU-задача. НЕ запускать на login-ноде с тяжёлым окружением — гонять в env lobs5
# (интерактивный compute-нод или sbatch), как run_beta.sh. Конвенция: results/ + logs/, timestamp.
#
#   bash run_methodology_pdf.sh                 # auto-движок (fpdf2 -> matplotlib fallback)
#   ENGINE=matplotlib bash run_methodology_pdf.sh   # форсить fallback (нулевые доп. deps)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p results logs
RUN_TS="$(date +%Y%m%d-%H%M%S)"
LOG="logs/methodology_pdf_${RUN_TS}.log"
OUT_TS="results/market_impact_methodology_${RUN_TS}.pdf"
OUT_STABLE="results/market_impact_methodology.pdf"
ENGINE="${ENGINE:-auto}"

{
  echo "=== methodology PDF build @ ${RUN_TS} (engine=${ENGINE}) ==="

  # conda env lobs5 (matplotlib/numpy), как в run_beta.sh — источаем conda.sh по известному пути
  CONDA_SH="${CONDA_SH:-/home/s5e/satyamaga.s5e/miniforge3/etc/profile.d/conda.sh}"
  if [ -r "$CONDA_SH" ]; then
    set +eu                       # conda.sh не -u/-e-safe
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "${CONDA_ENV:-lobs5}" || echo "WARN: conda activate ${CONDA_ENV:-lobs5} не удался — использую текущий python"
    set -eu
  else
    echo "WARN: conda.sh не найден ($CONDA_SH) — использую текущий python"
  fi
  echo "python: $(command -v python) ($(python --version 2>&1))"

  # fpdf2 — чистый python, без системных зависимостей; ставим в --user, если нет (и не форсим matplotlib)
  if [ "$ENGINE" != "matplotlib" ]; then
    python -c "import fpdf" 2>/dev/null || {
      echo "fpdf2 отсутствует — pip install --user fpdf2"
      python -m pip install --user fpdf2 || echo "WARN: pip fpdf2 не удался — будет matplotlib fallback"
    }
  fi

  JAX_PLATFORMS=cpu python make_methodology_pdf.py --engine "$ENGINE" --out "$OUT_TS"

  cp -f "$OUT_TS" "$OUT_STABLE"
  echo "PDF (timestamped): $HERE/$OUT_TS"
  echo "PDF (stable copy): $HERE/$OUT_STABLE"
  echo "=== done ==="
} 2>&1 | tee "$LOG"
