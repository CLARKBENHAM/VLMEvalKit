# gdt_vlmeval.py
import os
import argparse
import pandas as pd
import base64
from PIL import Image
import io
import re
from typing import Dict, List, Any, Union

# Import your existing functions
from hadrian_vllm.prompt_generator import element_ids_per_img_few_shot
from hadrian_vllm.utils import (
    load_csv,
    extract_assembly_and_page_from_filename,
    get_current_datetime,
)
from hadrian_vllm.evaluation import evaluate_answer
from hadrian_vllm.result_processor import extract_answer
from hadrian_vllm.model_caller import get_base64_image

# We need these for VLMEvalKit integration
from vlmeval.evaluate import Evaluator
from vlmeval.smp import build_model
from vlmeval.dataset.image_base import ImageBaseDataset


class GDTBenchmarkDataset(ImageBaseDataset):
    """Custom dataset class for GD&T benchmark evaluation"""

    TYPE = "GDT"
    DATASET_URL = None  # We'll load locally
    DATASET_MD5 = None  # Skip MD5 check

    def __init__(
        self,
        prompt_path: str = "/data2/Users/clark/hadrian_vllm/prompt6_claude_try_answers.txt",
        csv_path: str = "/data2/Users/clark/hadrian_vllm/data/fsi_labels/Hadrian Vllm test case - Final Merge.csv",
        eval_dir: str = "/data2/Users/clark/hadrian_vllm/data/eval_on/single_images/",
        n_shot_imgs: int = 6,
        eg_per_img: int = 50,
        multiturn: bool = True,
    ):
        super().__init__()
        self.prompt_path = prompt_path
        self.csv_path = csv_path
        self.eval_dir = eval_dir
        self.n_shot_imgs = n_shot_imgs
        self.eg_per_img = eg_per_img
        self.multiturn = multiturn
        assert self.multiturn, "wont implement single turn"

        # Create data file name based on configuration
        prompt_name = os.path.basename(prompt_path).replace(".txt", "")
        config_str = f"{prompt_name}_{n_shot_imgs}_{eg_per_img}_{'mt' if multiturn else 'st'}"
        self.data_file = f"data/datasets/gdt_benchmark_{config_str}.tsv"

        # Create the dataset if it doesn't exist
        if not os.path.exists(self.data_file):
            self._create_dataset()

    def _create_dataset(self):
        """Create the dataset TSV file"""
        print(
            "Creating GDT benchmark dataset with configuration: "
            f"{self.n_shot_imgs} shot images, {self.eg_per_img} examples per image, "
            f"{'multiturn' if self.multiturn else 'singleturn'}"
        )

        # Load the base CSV
        df = load_csv(self.csv_path)

        # Generate element_ids_by_image
        element_ids_by_image = {}
        for _, row in df.iterrows():
            if pd.isna(row["Assembly ID"]) or pd.isna(row["Page ID"]) or pd.isna(row["Element ID"]):
                continue

            assembly_id = int(row["Assembly ID"])
            page_id = int(row["Page ID"])
            element_id = row["Element ID"]

            # Find matching image file
            possible_path = None
            for file_name in os.listdir(self.eval_dir):
                if (
                    f"nist_ftc_{assembly_id:02d}" in file_name
                    and f"_pg{page_id}" in file_name
                    and file_name.endswith(".png")
                ):
                    possible_path = os.path.join(self.eval_dir, file_name)
                    break

            if possible_path:
                element_ids_by_image.setdefault(possible_path, []).append(element_id)

        # Create dataset samples
        samples = []
        for img_path, element_ids in element_ids_by_image.items():
            for element_id in element_ids:
                # Get ground truth
                assembly_id, page_id = extract_assembly_and_page_from_filename(img_path)
                row_mask = (
                    (df["Assembly ID"] == assembly_id)
                    & (df["Page ID"] == page_id)
                    & (df["Element ID"] == element_id)
                )

                specs = df.loc[row_mask, "Specification"].values
                if len(specs) == 0 or pd.isna(specs[0]):
                    continue

                ground_truth = specs[0]

                # Generate prompt with few-shot examples
                prompt_or_messages, image_paths, config = element_ids_per_img_few_shot(
                    text_prompt_path=self.prompt_path,
                    csv_path=self.csv_path,
                    eval_dir=self.eval_dir,
                    question_image=img_path,
                    question_ids=[element_id],
                    n_shot_imgs=self.n_shot_imgs,
                    eg_per_img=self.eg_per_img,
                    examples_as_multiturn=self.multiturn,
                )

                # Create sample
                sample = {
                    "index": len(samples),
                    "element_ids": [element_id],
                    "answer": [ground_truth],
                    "assembly_id": assembly_id,
                    "page_id": page_id,
                    "image_paths": image_paths,
                    "config": config,
                }

                # Store the prompt or messages
                if self.multiturn:
                    sample["messages"] = prompt_or_messages
                else:
                    sample["question"] = prompt_or_messages

                samples.append(sample)

        # Create the dataset file
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        pd.DataFrame(samples).to_csv(self.data_file, sep="\t", index=False)
        print(f"Created dataset with {len(samples)} samples at {self.data_file}")

    def build_prompt(self, line):
        """
        Build the prompt for a given sample.
        This is what VLMEvalKit will call when evaluating.
        """
        if isinstance(line, int):
            df = self.load()
            line = df.iloc[line]

        messages = eval(line["messages"])  # Convert string representation back to list
        # [dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]
        return [
            (
                [
                    dict(type="text", role=m["role"], value=m["content"]),
                    dict(type="image", role=m["role"], value=m["image_path"]),
                ]
                if "image_path" in m
                else [dict(type="text", role=m["role"], value=m["content"])]
            )
            for m in messages
        ]

        # # Format for VLMEvalKit
        # prompt_messages = []
        # for msg in messages:
        #     role = msg["role"]
        #     content = msg.get("content", "")

        #     if "image_path" in msg:
        #         # Message with image
        #         prompt_messages.append(
        #             {
        #                 "role": role,
        #                 "content": [
        #                     {"type": "image", "value": line["image"]},
        #                     {"type": "text", "value": content},
        #                 ],
        #             }
        #         )
        # return prompt_messages

    def evaluate(self, eval_file, **kwargs):
        """
        Evaluate model outputs.
        Uses your existing evaluate_answer function.
        """
        df = self.load(eval_file)

        # Evaluate each prediction
        results = []
        for _, row in df.iterrows():
            response = row["prediction"]
            ground_truth = row["answer"]

            preds = []
            for element_id in row["element_ids"]:
                model_pred = extract_answer(response, element_id)  # didn't test if could take list
                preds.append(model_pred)

        for model_pred, answer in zip(preds, ground_truth):
            try:
                score = evaluate_answer(model_pred, ground_truth)
                is_correct = score == 1.0
            except Exception as e:
                print(f"Error evaluating: {e}")
                is_correct = False

            results.append(
                {
                    "element_id": row["element_id"],
                    "correct": is_correct,
                    "prediction": extracted_answer,
                    "reference": ground_truth,
                }
            )

        # Calculate overall metrics
        results_df = pd.DataFrame(results)
        metrics = {
            "accuracy": results_df["correct"].mean() if len(results_df) > 0 else 0.0,
            "total_samples": len(results_df),
            "correct_samples": results_df["correct"].sum() if len(results_df) > 0 else 0,
        }

        # Per-assembly metrics (optional)
        if "assembly_id" in df.columns:
            for assembly_id in df["assembly_id"].unique():
                assembly_mask = df["assembly_id"] == assembly_id
                assembly_results = results_df[assembly_mask]
                if len(assembly_results) > 0:
                    metrics[f"accuracy_assembly_{assembly_id}"] = assembly_results["correct"].mean()

        return metrics


def run_evaluation(args):
    """Run evaluation with VLMEvalKit"""
    # Create the dataset
    dataset = GDTBenchmarkDataset(
        prompt_path=args.prompt,
        csv_path=args.csv_path,
        eval_dir=args.eval_dir,
        n_shot_imgs=args.n_shot_imgs,
        eg_per_img=args.eg_per_img,
        multiturn=args.multiturn,
    )

    # Create the evaluator
    evaluator = Evaluator(
        model=args.model,
        datasets=[dataset.TYPE],
        cached_file=None,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Run evaluation
    results = evaluator.evaluate()

    # Print results
    print("\nEvaluation Results:")
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GD&T models with VLMEvalKit")
    parser.add_argument("--prompt", required=True, help="Path to prompt template file")
    parser.add_argument(
        "--csv_path", default="data/fsi_labels/Hadrian Vllm test case - Final Merge.csv"
    )
    parser.add_argument("--eval_dir", default="data/eval_on/single_images/")
    parser.add_argument("--n_shot_imgs", type=int, default=3)
    parser.add_argument("--eg_per_img", type=int, default=5)
    parser.add_argument("--multiturn", action="store_true", help="Use multiturn format")
    parser.add_argument("--model", default="llava", help="Model name in VLMEvalKit")
    parser.add_argument("--model_path", help="Path/name for model initialization")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results")

    args = parser.parse_args()
    run_evaluation(args)
