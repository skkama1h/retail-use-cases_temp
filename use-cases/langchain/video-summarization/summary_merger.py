import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


class SummaryMerger:
    """
    Merge summaries generated from multiple chunks of text and generate a final summary with an anomaly score.
    """
    def __init__(self, model_id, device="CPU", max_new_tokens=512, batch_size=5):
        # openVINO configs for optimized model, apply uint8 quantization for lowering precision of key/value cache in LLMs.
        # apply dynamic quantization for activations
        ov_config = {"PERFORMANCE_HINT": "LATENCY",
                     "NUM_STREAMS": "1",
                     "CACHE_DIR": "./cache/ov_llama_cache",
                     "KV_CACHE_PRECISION": "u8",
                     "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
                     }
        # use langchain openVINO pipeline to load the model
        self.ov_llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            backend="openvino",
            model_kwargs={
                "device": device,
                "ov_config": ov_config,
                "trust_remote_code": True
            },
            pipeline_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "top_k": 10,
                "temperature": 0.7,
                "return_full_text": False,
                "repetition_penalty": 1.0,
                "encoder_repetition_penalty": 1.0
            })
        self.ov_llm.pipeline.tokenizer.pad_token_id = self.ov_llm.pipeline.tokenizer.eos_token_id

        template = """Write a response that appropriately completes the request. 
        ### Instruction: Please create a summary of the overall video highlighting all the important information. How would you rate the scene described on a scale from 0.0 to 1.0, with 0.0 representing a standard scene and 1.0 denoting a scene with suspicious activities? 
        Please organize your answer according to this example:

        **Overall Summary**: A summary of the entire text description in about five sentences or less.
        **Activity Observed**: Key actions observed in the video.
        **Potential Suspicious Activity**: List any activities that might indicate suspicious behavior.
        **Anomaly Score**: A number between 0.0 and 1.0 based on your analysis.

        ### Input: {question}
        ### Answer:"""

        self.prompt = PromptTemplate.from_template(template)
        # generation_config = {"skip_prompt": True, "pipeline_kwargs": {"max_new_tokens": max_new_tokens}}
        self.chain = self.prompt | self.ov_llm

        self.batch_size = batch_size

    def merge_summaries(self, summaries_file, out_file=None):
        print(f"Processing: {summaries_file}")

        with open(summaries_file, "r", encoding="utf-8") as handle:
            summaries = json.load(handle)

        start_time = time.time()
        chunks = list(summaries.values())

        num_batches = math.ceil(len(chunks) / self.batch_size)
        print(f"Num of batches to process: {num_batches}")

        batch_summaries = []
        # anomaly_scores = []
        # text = " ".join(summaries.values())

        for i in range(num_batches):
            print("--------------------------------------------")
            print(f"Processing batch {i + 1}...")
            batch_texts = chunks[i * self.batch_size:(i + 1) * self.batch_size]
            batch_summary = self.summarize_batch(batch_texts)
            batch_summaries.append(batch_summary)
            # anomaly_scores.append(self.extract_anomaly_score(batch_summary))

        # recursively merge summaries which are greater than batch size
        while len(batch_summaries) > self.batch_size:
            temp = []
            for i in range(0, len(batch_summaries), self.batch_size):
                group = batch_summaries[i: i + self.batch_size]
                temp.append(self.summarize_batch(group))
            batch_summaries = temp

        print("--------------------------------------------")
        print(f"Processing final batch of size {len(batch_summaries)}")
        # if multiple summaries are present, merge them, else use the single summary
        if len(batch_summaries) > 1:
            final_summary = self.summarize_batch(batch_summaries)
        else:
            final_summary = batch_summaries[0]
        # final_anomaly_score = sum(anomaly_scores) / len(anomaly_scores)

        # extract anomaly score from final summary using a regex pattern
        final_anomaly_score = self.extract_anomaly_score(final_summary)
        if out_file:
            with out_file.open("a") as handle:
                handle.write(f"\n\n{final_summary}")
                handle.write(f"\n\n**Final Anomaly Score**: {final_anomaly_score:.2f}")

        print("--------------------------------------------")
        print(final_summary)
        # print("--------------------------------------------")
        print(f"\n\n**Final Anomaly Score**: {final_anomaly_score:.2f}")

        # write overall summary and anomaly score to JSON file
        summaries["overall_summary"] = final_summary + f"\n\n**Final Anomaly Score**: {final_anomaly_score:.2f}"
        summaries["anomaly_score"] = f"{final_anomaly_score:.2f}"

        with open(summaries_file, "w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=4, ensure_ascii=False)

        print(f"Time taken for merge-summarize {summaries_file.name}: {time.time() - start_time:.2f} seconds")

    def summarize_batch(self, texts):
        text = " ".join(texts)
        merged = self.chain.invoke({"question": text})
        '''for chunk in self.chain.stream({"question": text}):
            # print(chunk, end="", flush=True)
            merged += chunk'''
        # print("\n")
        return merged.strip()

    @staticmethod
    def extract_anomaly_score(summary):
        # matching based on multiple scenarios observed; goal is to match floating point or integer after Anomaly Score
        # Anomaly Score sometimes is encapsulated within ** and sometimes LLM omits
        match = re.search(r"\*?\*?Anomaly Score\*?\*?:?\s*(-?\d+(\.\d+)?)", summary, re.DOTALL)
        if match:
            return float(match.group(1)) if match.group(1) else 0.0
        return 0.0


if __name__ == "__main__":
    description = "Generate merged summaries with an Anomaly Score indicating potential anomalies in the text summaries."
    parser = argparse.ArgumentParser(description)

    parser.add_argument("input_file", type=str,
                        help="Path to the chunk summaries file you want to summarize.")
    # optional model argument, uses lmware/llama-3.2-3b-instruct-ov by default
    parser.add_argument("-m", "--model_id", type=str,
                        help="Path to openvino-genai optimized model (local directory or HF id).",
                        default="llmware/llama-3.2-3b-instruct-ov")
    parser.add_argument("-d", "--device", type=str,
                        help="Target device for running the model.",
                        default="CPU")
    parser.add_argument("-t", "--max_new_tokens", type=int,
                        help="Maximum number of tokens to be generated.",
                        default=512)
    parser.add_argument("-b", "--batch_size", type=int,
                        help="Number of chunks to be proceeded in a batch",
                        default=5)
    parser.add_argument("-o", "--output_file", type=str,
                        help="File to write generated text")

    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print("Summaries file path does not exist. Please re-verify")
        exit()

    # optional argument to log console output to a file
    if args.output_file:
        output_file = Path(args.output_file)
        output_file.write_text(f"python {sys.argv}")

    # create instance of SummaryMerger class and merge summaries
    summary_merger = SummaryMerger(
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
    )

    summary_merger.merge_summaries(input_file, Path(args.output_file) if args.output_file else None)
