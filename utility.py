from tqdm import tqdm
from bert_score import score
from moverscore_v2 import word_mover_score
from collections import defaultdict
import random
import json

def evaluate_pipeline(dataset, pipeline):
    total_questions = 0
    mc_exact_count = 0
    mc_indirect_count = 0
    da_exact_count = 0
    da_indirect_count = 0
    total_bert_score = 0
    total_mover_score = 0
    predictions = {}

    for data in tqdm(dataset, desc="Evaluating", unit="sample", leave=False):
        image = data['image']
        question_id = data['question_id']
        question = data['question']
        choices = data['choices']
        true_mc_answer = data['choices'][data['correct_choice_idx']]
        true_da_answers = eval(data['direct_answers'])
        
        print("@" * 79)
        print("A-OKVQA question_id:", question_id)
        print("@" * 79)
        # Get the pipeline results
        tool_results, mc_answer, da_answer = pipeline(question, image, choices)
        
        print("-" * 79)
        print(f'Predicted Multiple Choice Answer: {mc_answer}')
        print(f'Predicted Direct Answer: {da_answer}')
        print("-" * 30)
        print(f'True Multiple Choice Answer: {true_mc_answer}')
        print(f'True Direct Answer: {true_da_answers}')
        print("-" * 79)

        # Save predictions
        predictions[question_id] = {
            'multiple_choice': mc_answer,
            'direct_answer': da_answer
        }

        # calculate Multiple Choice accuracy
        if true_mc_answer in mc_answer:
            mc_exact_count += 1
            print("Correct Multiple Choice answer! Exact match with predicted answer. Accuracy: +1")
        elif true_mc_answer in str(tool_results):
            mc_indirect_count += 1
            print("Correct Multiple Choice answer! Indirect match with tools' output. Accuracy: +1")
        else:
            print("Wrong. Multiple Choice answer is inaccurate. Accuracy: 0")

        # calculate Direct accuracy, BERTscore, MOVERscore
        if da_answer in true_da_answers:
            da_exact_count += 1
            print("Correct Direct answer! Exact match with predicted answer. Accuracy: +1")
        elif any(answer in str(tool_results) for answer in true_da_answers):
            da_indirect_count += 1
            print("Correct Direct answer! Indirect match with tools' output. Accuracy: +1")
        else:
            print("Wrong. Direct answer is inaccurate. Accuracy: 0")
        #BERTScore
        p, r, f1 = score([da_answer], [true_da_answers], lang='en')
        bert_score_val = f1.item()
        total_bert_score += bert_score_val
        print(f'BERTScore F1: {bert_score_val:.4f}')
        #MoverScore
        idf_dict = defaultdict(lambda: 1.)
        mover_scores = word_mover_score(true_da_answers, da_answer * len(true_da_answers), idf_dict, idf_dict, batch_size=1, device="cuda")
        mover_score_val = mover_scores[0]
        total_mover_score += mover_score_val
        print(f'MoverScore: {mover_score_val:.4f}')
        
        total_questions += 1
        
    # Calculate overall accuracy
    overall_mc_accuracy = (mc_exact_count + mc_indirect_count) / total_questions
    overall_da_accuracy = (da_exact_count + da_indirect_count) / total_questions
    average_bert_score = total_bert_score / total_questions
    average_mover_score = total_mover_score / total_questions

    # Display overall results
    print("=" * 79)
    print(f"Overall Multiple Choice Accuracy: {overall_mc_accuracy:.2f} ({(mc_exact_count / total_questions):.2f} exact + {(mc_indirect_count / total_questions):.2f} indirect)")
    print(f"Overall Direct Answer Accuracy: {overall_da_accuracy:.2f} ({(da_exact_count / total_questions):.2f} exact + {(da_indirect_count / total_questions):.2f} indirect)")
    print(f"Average Direct Answer BERTScore: {average_bert_score:.4f}")
    print(f"Average Direct Answer MoverScore: {average_mover_score:.4f}")
    print("=" * 79)

    filename = str(pipeline)[10:].split(' at')[0]
    random_number = random.randint(1, 1000000)
    
    # Save predictions in json file
    json_data = json.dumps(predictions, indent=4)
    with open(f"./logs/predictions_val_{filename}_{random_number}.json", "w") as f:
        f.write(json_data)

    # Save results to a file
    with open(f"./logs/results_{filename}_{random_number}.txt", "w") as f:
        f.write(f"Overall Multiple Choice Accuracy: {overall_mc_accuracy:.2f} ({(mc_exact_count / total_questions):.2f} exact + {(mc_indirect_count / total_questions):.2f} indirect)\n")
        f.write(f"Overall Direct Answer Accuracy: {overall_da_accuracy:.2f} ({(da_exact_count / total_questions):.2f} exact + {(da_indirect_count / total_questions):.2f} indirect)\n")
        f.write(f"Average Direct Answer BERTScore: {average_bert_score:.4f}\n")
        f.write(f"Average Direct Answer MoverScore: {average_mover_score:.4f}\n")
        
