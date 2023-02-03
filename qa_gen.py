import json

def convert_qa_text_to_dict(files=()):
    q_list = []
    for file in files:
        f = open(file, 'r', encoding='utf-8')
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
            elif "Select one:" in line or "Select one or more answer" in line or "Select all that apply" in line or "Select more than 1" in line or "Select One" in line or "Select one or more" in line:
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


def get_question_dict(p_question_text, p_choices_text, p_answer_text, p_explanation_text):
    new_q = {
        "question_text": ' '.join(p_question_text),  # separate each line by a single white space
        "choices_text": p_choices_text,  # retain list structure for easy later processing
        "answer_text": ''.join(p_answer_text),  # no need to add spaces for answer line
        "explanation_text": ' '.join(p_explanation_text),  # separate each line by a single white space
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

def further_fotmatting(my_qa_dict):
    for cur_old_q in my_qa_dict:
        # handling bugs in old qa dict first
        corrected_choices_text = []
        for choice in cur_old_q["choices_text"]:
            if "\u2028" in choice:
                corrected_choices_text += choice.split("\u2028")
            else:
                corrected_choices_text.append(choice)
        cur_old_q["choices_text"] = corrected_choices_text

        cur_new_q = {}

        # id section
        cur_new_q["id"] = ""

        # question section (including stem and choices)

        # stem
        cur_new_q["question"] = {}
        cur_new_q["question"]["stem"] = cur_old_q["question_text"]

        # choices
        cur_new_q["question"]["choices"] = []
        for choice in cur_old_q["choices_text"]:
            cur_choice_dict = {}
            cur_choice_dict["label"] = choice[0]
            cur_choice_dict["text"] = choice[3:]
            cur_new_q["question"]["choices"].append(cur_choice_dict)

        # answerKey section (should be only one word A, B, C, D, etc.)
        # currently explanation text are of no use, but I still factored it out 
        cur_new_q["answerKey"] = cur_old_q["answer_text"][0]
        if len(cur_old_q["answer_text"]) != 1:
            cur_old_q["explanation_text"] = cur_old_q["answer_text"][3:]

        # statements section (concatenate stem with choices text)
        cur_new_q["statements"] = []
        for choice in cur_new_q["question"]["choices"]:
            state = {"statement": cur_new_q["question"]["stem"]+" "+choice["text"]}
            cur_new_q["statements"].append(state)
        js_to_be_added = json.dumps(cur_new_q)
        with open("formatted_q.jsonl", "a") as outfile:
            print(js_to_be_added, file = outfile)



        

        





