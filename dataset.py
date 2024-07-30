from datasets import load_dataset
from IPython.display import display

def load_and_process_dataset(name):
    dataset = load_dataset(name)
    dataset = dataset['validation']
    
    return dataset

def get_samples(dataset, show=True):
    sample_x = {
        "image": dataset[1]['image'],
        "question": dataset[1]['question'],
        "choices": dataset[1]['choices'],
        "direct_answers": dataset[1]['direct_answers']
    }

    sample_y = {
        "image": dataset[2]['image'],
        "question": dataset[2]['question'],
        "choices": dataset[2]['choices'],
        "direct_answers": dataset[2]['direct_answers']
    }

    sample_z = {
        "image": dataset[3]['image'],
        "question": dataset[3]['question'],
        "choices": dataset[3]['choices'],
        "direct_answers": dataset[3]['direct_answers']
    }

    if show:
        print(f"sample_x:", "*" * 70)
        display(sample_x['image'])
        print(f"\tQuestion: {sample_x['question']}")
        print(f"\tChoices: {sample_x['choices']}")
        print(f"\tDirect Answers: {sample_x['direct_answers']}")

        print(f"sample_y:", "*" * 70)
        display(sample_y['image'])
        print(f"\tQuestion: {sample_y['question']}")
        print(f"\tChoices: {sample_y['choices']}")
        print(f"\tDirect Answers: {sample_y['direct_answers']}")

        print(f"sample_z:", "*" * 70)
        display(sample_z['image'])
        print(f"\tQuestion: {sample_z['question']}")
        print(f"\tChoices: {sample_z['choices']}")
        print(f"\tDirect Answers: {sample_z['direct_answers']}")

    return sample_x, sample_y, sample_z
