from pipeline import static_pipeline, dynamic_pipeline, noplan_pipeline
from utility import evaluate_pipeline

from datasets import load_dataset
import argparse
import os
import warnings
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate different pipelines on the dataset.")
    parser.add_argument("pipeline", type=str, choices=["static", "dynamic", "no_plan", "all", "debug"],
                        help="Choose which pipeline to evaluate: static, dynamic, no_plan, all, or debug.")
    parser.add_argument("portion", type=float, nargs='?', default=0.010,
                        help="Portion of dataset to use for the debug pipeline.")
    
    args = parser.parse_args()
    
    # Stop annoying warnings [run with caution]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.simplefilter('ignore')
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("bert_score").setLevel(logging.ERROR)
    
    # Load Dataset
    dataset = load_dataset("HuggingFaceM4/A-OKVQA", split='validation')
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
        # reduce data for debugging
        dataset = dataset.select(range(int(len(dataset) * args.portion)))
        
        print("Debugging static pipeline on three sample data...")
        evaluate_pipeline(dataset, static_pipeline)
        
        print("Debugging dynamic pipeline on three sample data...")
        evaluate_pipeline(dataset, dynamic_pipeline)
        
        print("Debugging noplan pipeline on three sample data...")
        evaluate_pipeline(dataset, noplan_pipeline)

    


