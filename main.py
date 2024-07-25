from dataset import load_and_process_dataset, get_samples
from pipeline import static_pipeline, dynamic_pipeline, noplan_pipeline
from utility import evaluate_pipeline
import os
import warnings
import logging


if __name__ == "__main__":
    # stop annoying warnings [run with caution]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.simplefilter('ignore')
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("bert_score").setLevel(logging.ERROR)
    
    dataset = load_and_process_dataset("HuggingFaceM4/A-OKVQA")
    print(dataset)
    
    #todo: image are not being displayed in terminal.
    # sample_x, sample_y, sample_z = get_samples(dataset)
    #todo: fix checking
    # # Check with a sample input
    # tool_results, mc_answer, da_answer = dynamic_pipeline(sample_z['question'], sample_z['image'], sample_z['choices'])
    # print("Multiple Choice Answer: \n", mc_answer)
    # print("Direct Answer: \n", da_answer)
    # tool_results, mc_answer, da_answer = static_pipeline(sample_z['question'], sample_z['image'], sample_z['choices'])
    # print("Multiple Choice Answer: \n", mc_answer)
    # print("Direct Answer: \n", da_answer)
    # tool_results, mc_answer, da_answer = noplan_pipeline(sample_z['question'], sample_z['image'], sample_z['choices'])
    # print("Multiple Choice Answer: \n", mc_answer)
    # print("Direct Answer: \n", da_answer)


    # Execute Pipeline Evaluation
    # evaluate_pipeline(dataset, static_pipeline)
    evaluate_pipeline(dataset, dynamic_pipeline)
    evaluate_pipeline(dataset, noplan_pipeline)

    




