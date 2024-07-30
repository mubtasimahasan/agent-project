from dataset import load_and_process_dataset, get_samples
from pipeline import static_pipeline, dynamic_pipeline, noplan_pipeline
from utility import evaluate_pipeline

import argparse
import os
import warnings
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate different pipelines on the dataset.")
    parser.add_argument("pipeline", type=str, choices=["static", "dynamic", "no_plan", "all", "debug"],
                        help="Choose which pipeline to evaluate: static, dynamic, no_plan, all, or debug.")
    
    args = parser.parse_args()
    
    print("\/" * 150)

    # Stop annoying warnings [run with caution]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.simplefilter('ignore')
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("bert_score").setLevel(logging.ERROR)
    
    # Load Dataset
    dataset = load_and_process_dataset("HuggingFaceM4/A-OKVQA")
    print(dataset)

    # Execute Pipeline Evaluation based on the choice
    if args.pipeline == "static":
        print("Evaluating dataset on static pipeline...")
        evaluate_pipeline(dataset, static_pipeline)
    
    elif args.pipeline == "dynamic":
        print("Evaluating dataset on dynamic pipeline...")
        evaluate_pipeline(dataset, dynamic_pipeline)
    
    elif args.pipeline == "no_plan":
        print("Evaluating dataset on no-plan pipeline...")
        evaluate_pipeline(dataset, noplan_pipeline)
    
    elif args.pipeline == "all":
        print("Evaluating dataset on static pipeline...")
        evaluate_pipeline(dataset, static_pipeline)
        
        print("Evaluating dataset on dynamic pipeline...")
        evaluate_pipeline(dataset, dynamic_pipeline)
        
        print("Evaluating dataset on no-plan pipeline...")
        evaluate_pipeline(dataset, noplan_pipeline)
    
    elif args.pipeline == "debug":
        print("Debugging pipelines on three sample data...")
        sample_x, sample_y, sample_z = get_samples(dataset)

        tool_results, mc_answer, da_answer = static_pipeline(sample_z['question'], sample_z['image'], sample_z['choices'])
        print("Multiple Choice Answer: \n", mc_answer)
        print("Direct Answer: \n", da_answer)
        
        tool_results, mc_answer, da_answer = dynamic_pipeline(sample_z['question'], sample_z['image'], sample_z['choices'])
        print("Multiple Choice Answer: \n", mc_answer)
        print("Direct Answer: \n", da_answer)
        
        tool_results, mc_answer, da_answer = noplan_pipeline(sample_z['question'], sample_z['image'], sample_z['choices'])
        print("Multiple Choice Answer: \n", mc_answer)
        print("Direct Answer: \n", da_answer)

    


