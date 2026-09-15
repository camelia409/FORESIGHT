"""
pipeline.py — End-to-End Pipeline Orchestration
================================================
Project: FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

Responsibilities
----------------
- Provide a single entry-point to run the full FORESIGHT pipeline.
- Orchestrate stages in order: ingest → validate → preprocess → ... → risk.
- Phase 1A implements: ingest, validate (with raw hash integrity).
- Subsequent phases add: preprocess, aggregate, features, baseline, train,
  backtest, forecast, risk.

Phase 1A Pipeline (this implementation)
----------------------------------------
1. Load configuration
2. Compute raw file SHA-256 hashes (before)
3. Load all four raw datasets
4. Run all validation checks
5. Generate validation report (markdown + CSV)
6. Compute raw file SHA-256 hashes (after)
7. Assert raw hashes unchanged
8. Exit non-zero only on genuine FAIL conditions

Design Principles
-----------------
- No global mutable state — every pipeline run is stateless.
- All parameters come from configs/config.yaml via src.config.
- Known policy warnings (orphan SKUs, negative margin, IV basis) do NOT
  cause a non-zero exit — they are logged and reported.
- CRITICAL validation failures cause a non-zero exit.
- The pipeline never writes to data/raw/.

CLI usage
---------
    python -m src.pipeline             # run Phase 1A (ingest + validate)
    python -m src.pipeline --phase 1   # explicit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import CFG, PATHS
from src.utils import setup_logging, hash_raw_files, assert_hashes_unchanged

logger = logging.getLogger("foresight.pipeline")

# Pipeline stage names in execution order
PIPELINE_STAGES = [
    "ingest",
    "validate",
    "preprocess",
    "aggregate",
    "features",
    "baseline",
    "train",
    "backtest",
    "forecast",
    "risk",
]


class ForesightPipeline:
    """
    End-to-end FORESIGHT pipeline orchestrator.

    Stages that are not yet implemented raise NotImplementedError, which is
    caught and logged as 'not_implemented' — the pipeline does not crash
    during the skeleton phase.

    Phase 1A stages (ingest, validate) are fully implemented.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._run_id  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._log: dict = {
            "run_id":         self._run_id,
            "project":        CFG.project_name,
            "version":        CFG.project_version,
            "phases_done":    [],
            "stages":         {},
            "raw_hashes_before": {},
            "raw_hashes_after":  {},
            "hash_integrity":    None,
        }
        self._data:       dict = {}
        self._suite:      object = None  # ValidationSuite set after validate stage
        logger.info("ForesightPipeline initialised. run_id=%s  version=%s",
                    self._run_id, CFG.project_version)

    # -------------------------------------------------------------------------
    # Stage dispatcher
    # -------------------------------------------------------------------------

    def _run_stage(self, name: str) -> bool:
        """
        Execute a single named stage.
        Returns True if completed; False if skipped (NotImplemented).
        Raises on genuine errors.
        """
        logger.info("=" * 64)
        logger.info("STAGE: %-12s [run_id=%s]", name.upper(), self._run_id)
        t0 = time.time()
        try:
            method = getattr(self, f"_stage_{name}")
            method()
            elapsed = round(time.time() - t0, 3)
            self._log["stages"][name] = {"status": "completed", "elapsed_s": elapsed}
            logger.info("STAGE %-12s DONE in %.3fs", name, elapsed)
            return True
        except NotImplementedError as exc:
            self._log["stages"][name] = {"status": "not_implemented", "note": str(exc)}
            logger.warning("STAGE %-12s — NOT YET IMPLEMENTED: %s", name, exc)
            return False
        except Exception as exc:
            self._log["stages"][name] = {"status": "error", "error": str(exc)}
            logger.error("STAGE %-12s — ERROR: %s", name, exc, exc_info=True)
            raise

    def run(self, stages: str | list[str] = "all") -> dict:
        """
        Run pipeline stages.

        Parameters
        ----------
        stages : "all" or a list of stage names.

        Returns
        -------
        dict — run summary log.
        """
        if stages == "all":
            stages_to_run = PIPELINE_STAGES
        else:
            stages_to_run = stages

        for s in stages_to_run:
            if s not in PIPELINE_STAGES:
                raise ValueError(f"Unknown stage '{s}'. Valid stages: {PIPELINE_STAGES}")

        logger.info("Starting ForesightPipeline run_id=%s  stages=%s",
                    self._run_id, stages_to_run)

        for stage in stages_to_run:
            self._run_stage(stage)

        logger.info("Pipeline complete. run_id=%s  summary=%s",
                    self._run_id, self._log["stages"])
        return self._log

    # -------------------------------------------------------------------------
    # Stage implementations
    # -------------------------------------------------------------------------

    def _stage_ingest(self) -> None:
        """
        Phase 1A — Stage 1: Load all four raw datasets.

        - Computes SHA-256 hashes BEFORE loading (integrity baseline).
        - Loads via data_ingestion.load_all_data().
        - Verifies required columns at load time.
        - Stores datasets in self._data for downstream stages.
        """
        from src.data_ingestion import load_all_data

        logger.info("[ingest] Computing raw file hashes (before load)…")
        self._log["raw_hashes_before"] = hash_raw_files(PATHS.raw_dir)

        logger.info("[ingest] Loading raw datasets…")
        self._data = load_all_data()

        logger.info(
            "[ingest] Loaded: sales=%d  sku=%d  calendar=%d  inventory=%d",
            len(self._data["sales"]),
            len(self._data["sku"]),
            len(self._data["calendar"]),
            len(self._data["inventory"]),
        )

    def _stage_validate(self) -> None:
        """
        Phase 1A — Stage 2: Run all validation checks.

        - Validates schema, primary keys, domain constraints, RI, panel.
        - Produces a ValidationSuite with PASS / WARNING / FAIL results.
        - Saves validation report (markdown) and CSV artefact.
        - Computes SHA-256 hashes AFTER to verify raw files unchanged.
        - Raises ValidationError only on CRITICAL FAIL (blocks pipeline).
        - WARNING results (orphan SKUs, neg margin, IV basis) do NOT fail.
        """
        from src.validation import run_all_validations, save_validation_report, ValidationError

        if not self._data:
            raise RuntimeError("[validate] No data available. Run 'ingest' stage first.")

        logger.info("[validate] Running validation suite…")
        suite = run_all_validations(self._data)
        self._suite = suite

        # Save reports
        logger.info("[validate] Saving validation report…")
        md_path, csv_path = save_validation_report(suite)

        # Log summary
        logger.info("[validate] %s", suite.summary_line())
        logger.info("[validate] Markdown report: %s", md_path)
        logger.info("[validate] CSV artefact   : %s", csv_path)

        # Verify raw files unchanged AFTER validation
        logger.info("[validate] Verifying raw file integrity (after validation)…")
        self._log["raw_hashes_after"] = hash_raw_files(PATHS.raw_dir)
        try:
            assert_hashes_unchanged(
                self._log["raw_hashes_before"],
                self._log["raw_hashes_after"],
            )
            self._log["hash_integrity"] = "PASS"
            logger.info("[validate] Raw file integrity: PASS — all files unchanged.")
        except RuntimeError as exc:
            self._log["hash_integrity"] = "FAIL"
            logger.critical("[validate] RAW FILE INTEGRITY VIOLATION: %s", exc)
            raise

        # Update log
        self._log["validation"] = {
            "n_pass":    suite.n_pass,
            "n_warning": suite.n_warning,
            "n_fail":    suite.n_fail,
            "has_critical": suite.has_critical,
            "report_md":  str(md_path),
            "report_csv": str(csv_path),
        }

        # Raise on CRITICAL FAILs only (known warnings are not critical)
        if suite.has_critical:
            critical = [r for r in suite.results if r.status == "FAIL" and r.severity == "CRITICAL"]
            msg = "\n".join(f"  [{r.dataset}] {r.check_name}: {r.message}" for r in critical)
            raise ValidationError(
                f"[validate] {len(critical)} CRITICAL validation failure(s):\n{msg}"
            )

        if suite.n_fail > 0:
            fail_results = [r for r in suite.results if r.status == "FAIL"]
            logger.error(
                "[validate] %d non-critical FAIL check(s). Review report: %s",
                suite.n_fail, md_path,
            )

    def _stage_preprocess(self) -> None:
        """
        Phase 1B — Preprocessing, Quarantine & Data Integration.

        - Normalises sales, sku_master, calendar, and inventory.
        - Quarantines 150 orphan inventory SKUs (SKU051–SKU200, 3,600 rows).
        - Safely integrates master datasets on SKU + Date grain (exactly 36,550 rows).
        - Verifies all 16 Phase 1B Data Quality Invariants.
        - Persists interim and processed datasets, data dictionary, report, and lineage.
        - Verifies raw CSV files remain unchanged.
        """
        from src.preprocessing import build_analysis_ready, save_analysis_ready

        if not self._data:
            raise RuntimeError("[preprocess] No data available. Run 'ingest' stage first.")

        logger.info("[preprocess] Starting Phase 1B normalisation & integration…")
        analysis_ready_df, interim_master_df, inv_quarantine_df, inv_results = build_analysis_ready(
            sales=self._data["sales"],
            sku=self._data["sku"],
            cal=self._data["calendar"],
            inv=self._data["inventory"],
            raw_dir=PATHS.raw_dir,
            raw_hashes_before=self._log["raw_hashes_before"],
        )

        # Store datasets in pipeline memory
        self._data["analysis_ready"] = analysis_ready_df
        self._data["interim_master"] = interim_master_df
        self._data["inventory_quarantine"] = inv_quarantine_df

        # Save artifacts
        saved_paths = save_analysis_ready(
            analysis_ready_df=analysis_ready_df,
            interim_master_df=interim_master_df,
            inventory_quarantine_df=inv_quarantine_df,
            invariant_results=inv_results,
            raw_hashes_before=self._log["raw_hashes_before"],
            raw_hashes_after=self._log.get("raw_hashes_after"),
        )

        # Verify raw integrity once more after preprocessing
        logger.info("[preprocess] Verifying raw file integrity after preprocessing…")
        current_hashes = hash_raw_files(PATHS.raw_dir)
        assert_hashes_unchanged(self._log["raw_hashes_before"], current_hashes)

        # Update run log
        self._log["preprocessing"] = {
            "analysis_ready_rows": len(analysis_ready_df),
            "analysis_ready_skus": int(analysis_ready_df["SKU"].nunique()),
            "analysis_ready_dates": int(analysis_ready_df["Date"].nunique()),
            "quarantine_rows": len(inv_quarantine_df),
            "quarantine_skus": int(inv_quarantine_df["SKU"].nunique()),
            "invariants_passed": sum(1 for r in inv_results.values() if r["status"] == "PASS"),
            "invariants_total": len(inv_results),
            "artifacts": {k: str(v) for k, v in saved_paths.items()},
        }
        logger.info(
            "[preprocess] Preprocessing & integration complete. Rows: %d, Quarantine: %d, Invariants: %d/%d PASS",
            len(analysis_ready_df),
            len(inv_quarantine_df),
            self._log["preprocessing"]["invariants_passed"],
            self._log["preprocessing"]["invariants_total"],
        )

    def _stage_aggregate(self) -> None:
        raise NotImplementedError("aggregate stage — implement in Phase 1B.")

    def _stage_features(self) -> None:
        raise NotImplementedError("features stage — implement in Phase 2.")

    def _stage_baseline(self) -> None:
        raise NotImplementedError("baseline stage — implement in Phase 3.")

    def _stage_train(self) -> None:
        raise NotImplementedError("train stage — implement in Phase 4.")

    def _stage_backtest(self) -> None:
        raise NotImplementedError("backtest stage — implement in Phase 5.")

    def _stage_forecast(self) -> None:
        raise NotImplementedError("forecast stage — implement in Phase 6.")

    def _stage_risk(self) -> None:
        raise NotImplementedError("risk stage — implement in Phase 6.")

    def get_run_summary(self) -> dict:
        """Return the current run's log dictionary."""
        return dict(self._log)

    def _save_run_manifest(self) -> None:
        """Write pipeline run manifest to artifacts/metrics/."""
        manifest_dir = PATHS.metrics_dir
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"pipeline_run_{self._run_id}.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(self._log, fh, indent=2, default=str)
        logger.info("[pipeline] Run manifest saved: %s", manifest_path)


# ---------------------------------------------------------------------------
# Phase 1A entry point (run only ingest + validate)
# ---------------------------------------------------------------------------

def run_phase1(verbose: bool = False) -> int:
    """
    Execute Phase 1: Ingestion, Validation (1A) + Preprocessing & Integration (1B).

    Returns
    -------
    int — exit code (0 = success, 1 = critical failure).
    """
    from src.validation import ValidationError

    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=log_level, name="foresight")

    logger.info("=" * 64)
    logger.info("  PROJECT FORESIGHT — Phase 1: Ingestion, Validation & Integration")
    logger.info("  orphan_sku_treatment    : %s", CFG.orphan_sku_treatment)
    logger.info("  negative_margin_treatment: %s", CFG.negative_margin_treatment)
    logger.info("  inventory_valuation_basis: %s", CFG.inventory_valuation_basis)
    logger.info("  random_seed             : %d", CFG.random_seed)
    logger.info("=" * 64)

    pipeline = ForesightPipeline()
    exit_code = 0

    try:
        pipeline.run(stages=["ingest", "validate", "preprocess"])
    except (ValidationError, RuntimeError) as exc:
        logger.critical("Phase 1 FAILED: %s", exc)
        exit_code = 1
    except Exception as exc:
        logger.critical("Unexpected error in Phase 1: %s", exc, exc_info=True)
        exit_code = 1

    pipeline._save_run_manifest()

    summary = pipeline.get_run_summary()
    val     = summary.get("validation", {})
    prep    = summary.get("preprocessing", {})

    print("\n" + "=" * 64)
    print("  Phase 1 Execution Summary")
    print("=" * 64)
    for stage, info in summary.get("stages", {}).items():
        status = info.get("status", "?")
        print(f"  {stage:<14s}: {status}")

    print(f"\n  Validation Results (Phase 1A)")
    print(f"    PASS    : {val.get('n_pass', '?')}")
    print(f"    WARNING : {val.get('n_warning', '?')}")
    print(f"    FAIL    : {val.get('n_fail', '?')}")

    print(f"\n  Preprocessing & Integration (Phase 1B)")
    print(f"    Analysis Ready Rows   : {prep.get('analysis_ready_rows', '?')} (expected 36,550)")
    print(f"    Analysis Ready SKUs   : {prep.get('analysis_ready_skus', '?')} (expected 50)")
    print(f"    Analysis Ready Dates  : {prep.get('analysis_ready_dates', '?')} (expected 731)")
    print(f"    Quarantined Rows      : {prep.get('quarantine_rows', '?')} (expected 3,600)")
    print(f"    Quarantined SKUs      : {prep.get('quarantine_skus', '?')} (expected 150)")
    print(f"    Invariants Passed     : {prep.get('invariants_passed', '?')} / {prep.get('invariants_total', '?')}")

    print(f"\n  Raw Hash Integrity : {summary.get('hash_integrity', '?')}")
    print(f"\n  Validation Report   : {val.get('report_md',  '?')}")
    print(f"  Validation CSV      : {val.get('report_csv', '?')}")
    if prep.get("artifacts"):
        print(f"  Processed Parquet   : {prep['artifacts'].get('analysis_ready', '?')}")
        print(f"  Interim Master      : {prep['artifacts'].get('interim_master', '?')}")
        print(f"  Inventory Quarantine: {prep['artifacts'].get('inventory_quarantine', '?')}")
        print(f"  Data Dictionary     : {prep['artifacts'].get('data_dictionary', '?')}")
        print(f"  Preprocessing Report: {prep['artifacts'].get('report', '?')}")
        print(f"  Lineage Manifest    : {prep['artifacts'].get('lineage', '?')}")
    print("=" * 64)

    if exit_code == 0:
        print("  Phase 1 (1A + 1B): SUCCESS")
    else:
        print("  Phase 1: FAILED — see log for details")
    print("=" * 64)

    return exit_code


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PROJECT FORESIGHT Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1],
        help="Pipeline phase to execute (currently only 1 is implemented).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.phase == 1:
        sys.exit(run_phase1(verbose=args.verbose))
    else:
        print(f"Phase {args.phase} is not yet implemented.")
        sys.exit(1)
