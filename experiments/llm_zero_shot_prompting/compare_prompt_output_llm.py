"""
Check that the LLM does not hallucinate new hypothesis
"""
import os
import click
import pandas as pd
from tqdm import tqdm
from collections import Counter

COMPARATIVE_VAL = ["higher", "lower", "Higher", "Lower"]
OUTPUT_LEAVE_OUT = ["dependent", "comparative"]

def prep(output_p, prompt_p):
    """ Arrange data """
    output = pd.read_csv(output_p, index_col=0)
    prompt = pd.read_csv(prompt_p, index_col=0)

    if all(x in COMPARATIVE_VAL for x in output.comparative.unique()):
        comparative = "comp_ok"
    else:
        comparative = "comp_ko"

    output = output[[x for x in output.columns if x not in OUTPUT_LEAVE_OUT]]
    prompt = prompt[output.columns]
    for col in prompt.columns:
        prompt[col] = prompt[col].apply(lambda x: x.split('/')[-1] if isinstance(x, str) else x)

    for col in output.columns:
        output[col] = output[col].astype(str).str.lower()
        prompt[col] = prompt[col].astype(str).str.lower()
    return output, prompt, comparative

def compare_one(output_p, prompt_p):
    """ Main """
    output, prompt, comparative = prep(output_p=output_p, prompt_p=prompt_p)

    check_rows = output.apply(lambda row: (prompt == row).all(axis=1).any(), axis=1)
    if check_rows.all():
        return comparative, "same"
    return comparative, "different"

def compare_all(folder):
    """ Compare all to compare in folder """
    outputs = os.path.join(folder, "outputs")
    prompts = os.path.join(folder, "prompts")

    outputs_csv = [x for x in os.listdir(outputs) if x.endswith(".csv")]
    outcomes, comparatives, different, comp = [], [], [], []
    for o in tqdm(outputs_csv):
        comparative, outcome = compare_one(output_p=os.path.join(outputs, o), prompt_p=os.path.join(prompts, o))
        outcomes.append(outcome)
        comparatives.append(comparative)
        if outcome == "different":
            different.append(o)
        if comparative == "comp_ko":
            comp.append(o)
    
    print(Counter(outcomes))
    print(Counter(comparatives))
    return different, comp


def inspect_different(folder, name):
    """ Inspect how many not in the original prompt """
    output, prompt, _ = prep(
        output_p=os.path.join(folder, "outputs", name),
        prompt_p=os.path.join(folder, "prompts", name))
    check_rows = output.apply(lambda row: (prompt == row).all(axis=1).any(), axis=1)
    print(name, (~check_rows).sum(), output.shape[0])


@click.command()
@click.argument("folder")
def main(folder):
    """ Going through all files in the sub-folders """
    different, comp = compare_all(folder=folder)
    print(f"Files where the `comparative` is not in ['higher', 'lower']: {comp}")
    for name in tqdm(different):
        inspect_different(folder, name)


if __name__ == '__main__':
    #`comparative` is not in ["higher", "lower"] -> does not hallucinate but outputs the templated sentence instead of the .csv output
    # python experiments/llm_zero_shot_prompting/compare_prompt_output_llm.py experiments/llm_zero_shot_prompting/final/h_regular_es_d
    # python experiments/llm_zero_shot_prompting/compare_prompt_output_llm.py experiments/llm_zero_shot_prompting/final/h_study_mod_es_d
    # python experiments/llm_zero_shot_prompting/compare_prompt_output_llm.py experiments/llm_zero_shot_prompting/final/h_var_mod_es_d
    main()