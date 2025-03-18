from openai import OpenAI
from shared.const import task_rel_labels, task_ner_labels
import pprint

datasets = task_ner_labels.keys()
llm = 'o1'

def query_llm(openai_key, llm, message):
    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model=llm,
        messages=message,
        max_completion_tokens=4096,
        seed=0,
    )
    return response

def generate_description(task, openai_key):
    ners = task_ner_labels[task]
    rels = task_rel_labels[task]
    descriptions = {}

    if len(ners) > 2:
        descriptions[0] = [" there are no relations between the @subject@ and the @object@ ."]
        for i, rel in enumerate(rels):
            system_content = f'Explain the {rel} relationship between a subject entity and an object entity by crafting a sentence that includes the placeholders @subject@ and @object@.'

            user = {'role': 'user',
                    'content': system_content}

            messages = [user]

            response = query_llm(openai_key, llm, messages)
            # print('content', response)
            descriptions[i+1] = [response.choices[0].message.content]

    else:
        e1_type = ners[0]
        e2_type = ners[1]
        descriptions[0] = [f" there are no relations between the {e1_type} @subject@ and the {e2_type} @object@ ."]

        for i, rel in enumerate(rels):
            system_content = f'Explain the {rel} relationship between a {e1_type} (subject) and a {e2_type} (object) by crafting a sentence that includes the placeholders @subject@ and @object@.'

            user = {'role': 'user',
                    'content': system_content}

            messages = [user]
            response = query_llm(openai_key, llm, messages)
            # print('content', response.choices[0].message.content)
            descriptions[i + 1] = [response.choices[0].message.content]

    return descriptions

def dump_dict_to_py_file(dictionary, filename):

    with open(filename, "w") as file:
        file.write("# The discriptions were automatically generated.\n\n")
        file.write("descriptions = ")
        file.write(pprint.pformat(dictionary))
        file.write("\n")

if __name__ == "__main__":
    openai_key = '<openai_key>'
    generated_descriptions = {}
    for task in datasets:
        generated_descriptions[task] = generate_description(task, openai_key)
    # dump to .py file for easy read.
    dump_dict_to_py_file(generated_descriptions, 'shared/generated_descriptions.py')
