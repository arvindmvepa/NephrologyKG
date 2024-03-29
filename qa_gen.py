import os
from sklearn.model_selection import train_test_split


def create_train_dev_test_split(base_file, nephqa_root=r"C:\Users\fabien\Documents\Arvind\GreaseLM\data\nephqa",
                                test_prop=0.15, dev_prop=0.10, random_state=0):
    with open(base_file) as f:
        lines = f.readlines()
    traindev, test = train_test_split(lines, test_size=test_prop, random_state=random_state)
    train, dev = train_test_split(traindev, test_size=dev_prop, random_state=random_state)

    statement_dir = os.path.join(nephqa_root, "statement")
    if not os.path.exists(statement_dir):
        os.makedirs(statement_dir)

    train_file = os.path.join(statement_dir, "train.statement.jsonl")
    dev_file = os.path.join(statement_dir, "dev.statement.jsonl")
    test_file = os.path.join(statement_dir, "test.statement.jsonl")

    with open(train_file, "w") as f:
        f.writelines(train)
    with open(dev_file, "w") as f:
        f.writelines(dev)
    with open(test_file, "w") as f:
        f.writelines(test)

    return {"train_file": train_file, "dev_file": dev_file, "test_file": test_file}


def convert_qa_text_to_dict(files=()):
    q_list = []
    for file in files:
        f = open(file, 'r')
        on_question, on_choices, on_answer, on_explanation = set_question_part("start")
        for line in f:
            # ignore lines of white space
            if line.isspace():
                continue
            if "Question" in line:
                # store question after getting to the next question
                if on_explanation:
                    new_q = get_question_dict(question_text, choices_text, answer_text, explanation_text)
                    q_list.append(new_q)
                question_text = []
                on_question, on_choices, on_answer, on_explanation = set_question_part("question")
            elif "Select one:" in line:
                choices_text = []
                on_question, on_choices, on_answer, on_explanation = set_question_part("choices")
            elif "Answer" in line:
                answer_text = []
                on_question, on_choices, on_answer, on_explanation = set_question_part("answer")
            elif on_question:
                question_text.append(line.strip())
            elif on_choices:
                choices_text.append(line.strip())
            elif on_answer:
                answer_text.append(line.strip())
                on_question, on_choices, on_answer, on_explanation = set_question_part("explanation")
                explanation_text = []
            elif on_explanation:
                explanation_text.append(line.strip())
        # make sure to store last question
        if on_explanation:
            new_q = get_question_dict(question_text, choices_text, answer_text, explanation_text)
            q_list.append(new_q)
    return q_list


def get_question_dict(question_text, choices_text, answer_text, explanation_text):
    new_q = {
        "question_text": ' '.join(question_text),  # separate each line by a single white space
        "choices_text": choices_text,  # retain list structure for easy later processing
        "answer_text": ''.join(answer_text),  # no need to add spaces for answer line
        "explanation_text": ' '.join(explanation_text),  # separate each line by a single white space
    }
    return new_q


def set_question_part(part):
    if part == "start":
        return False, False, False, False
    if part == "question":
        return True, False, False, False
    elif part == "choices":
        return False, True, False, False
    elif part == "answer":
        return False, False, True, False
    elif part == "explanation":
        return False, False, False, True
    else:
        raise ValueError(f"question part {part} is not recognized")





