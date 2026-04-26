from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adult_lab.doc_report import write_doc_report
from adult_lab.pipeline import download_data, evaluate_model, profile_data, run_all, train_model


DEFAULT_CONFIG_PATH = Path("configs/default.toml")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adult dataset decision tree and random forest experiment CLI"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the TOML configuration file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download-data", help="Download Adult raw data files if needed.")
    subparsers.add_parser("profile-data", help="Profile raw and processed datasets.")

    train_parser = subparsers.add_parser("train", help="Train a specific model.")
    train_parser.add_argument("--model", choices=["tree", "forest"], required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a previously trained model.")
    evaluate_parser.add_argument("--model", choices=["tree", "forest"], required=True)

    doc_parser = subparsers.add_parser("write-doc-report", help="Generate a DOCX experiment report from the template.")
    doc_parser.add_argument("--template", default="学生实验报告.docx", help="Path to the DOCX template.")
    doc_parser.add_argument(
        "--output",
        default="reports/实验四_决策树与随机森林分类_实验报告.docx",
        help="Path to the output DOCX report.",
    )
    doc_parser.add_argument("--student-name", default="（请填写）")
    doc_parser.add_argument("--student-id", default="（请填写）")
    doc_parser.add_argument("--class-name", default="（请填写）")
    doc_parser.add_argument("--report-date", default="")

    subparsers.add_parser("run-all", help="Run the complete experiment pipeline.")
    return parser


def _print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "download-data":
        _print_payload(download_data(args.config))
        return 0
    if args.command == "profile-data":
        _print_payload(profile_data(args.config))
        return 0
    if args.command == "train":
        _print_payload(train_model(args.config, args.model))
        return 0
    if args.command == "evaluate":
        _print_payload(evaluate_model(args.config, args.model))
        return 0
    if args.command == "write-doc-report":
        _print_payload(
            write_doc_report(
                config_path=args.config,
                template_path=args.template,
                output_path=args.output,
                student_name=args.student_name,
                student_id=args.student_id,
                class_name=args.class_name,
                report_date=args.report_date,
            )
        )
        return 0
    if args.command == "run-all":
        _print_payload(run_all(args.config))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
