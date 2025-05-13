import json
import logging
from typing import Dict, List

from sklearn.metrics import roc_auc_score
from tabulate import tabulate  # pip install tabulate

# 模型和数据集名称
model_names = ["gpt-4o-mini", "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"]
dataset_names = ["gsm8k", "medqa_us", "triviaqa"]

# 设置日志格式
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_auroc(results: List[Dict]) -> float:
    """从结果中计算 AUROC"""
    confidence_list = []
    is_correct_list = []
    for item in results:
        if 0 <= item["confidence"] <= 100:
            confidence_list.append(item["confidence"] / 100)
            is_correct_list.append(item["is_correct"])
    return roc_auc_score(is_correct_list, confidence_list)


def main():
    for model_name in model_names:
        logger.info("=" * 100)
        logger.info(f"{model_name:^100}")
        logger.info("=" * 100)

        table_data = []
        headers = ["Dataset", "Calibrate", "Reported AUROC", "Computed AUROC"]

        for dataset_name in dataset_names:
            for calibrate in ["True", "False"]:
                path = f"results/{model_name}/{dataset_name}/results_calibrate_{calibrate}.json"
                try:
                    with open(path, "r") as f:
                        results = json.load(f)
                    reported_auroc = results.get("auroc", "N/A")
                    computed_auroc = compute_auroc(results["data"])
                    table_data.append(
                        [
                            dataset_name,
                            calibrate,
                            (
                                f"{reported_auroc:.4f}"
                                if isinstance(reported_auroc, float)
                                else reported_auroc
                            ),
                            f"{computed_auroc:.4f}",
                        ]
                    )
                except FileNotFoundError:
                    table_data.append(
                        [dataset_name, calibrate, "File Not Found", "N/A"]
                    )
                except Exception as e:
                    table_data.append([dataset_name, calibrate, f"Error: {e}", "N/A"])

        logger.info(tabulate(table_data, headers=headers, tablefmt="grid"))
        logger.info("\n")


if __name__ == "__main__":
    main()
