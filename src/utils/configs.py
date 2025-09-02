"""This module contains the configurations for getting
inputs for the models.
"""
from typing import Text, Dict, Any, List, Union, Optional, Iterable
from tqdm import tqdm
from collections import Counter
import json
import pandas as pd
import numpy as np
import os
import re
import torch.distributed as dist
from torch.utils.data import (
    DataLoader
)

from src.models.model import (
    OpenAIModel,
    AnthropicModel,
    OpenModel,
    OpenModelVLLM,
    OpenModelHF,
)
from src.collate_fns.codeforce_collate_fn import (
    DPGenerateCollateFn,
    CodeforceDPInferenceCollateFn
)
from src.dp.dataset import (
    CodeforceDPGenerateDataset,
    CodexDPGenerateDataset,
    DPInferenceDataset
)
from src.dp.dp_generator import (
    APIModelSingleThreadDPGenerator,
    APIModelParallelThreadDPInference,
    OpenModelParallelThreadDPInference
)

from src.evaluators.dp_evaluator import (
    CodeForceCorrectnessEvaluator,
    CodexCorrectnessEvaluator
)

from src.evaluators.evaluation_utils import (
    write_json,
    combine_results
)

from src.dp.generator import TECHNIQUES

from dotenv import load_dotenv
load_dotenv()

__MODEL_TO_CLASS__ = {
    "gpt-5-mini": OpenAIModel,
    "gpt-4-1106-preview": OpenAIModel,
    "gpt-4": OpenAIModel,
    "gpt-3.5-turbo": OpenAIModel,
    "anthropic.claude-3-sonnet-20240229-v1:0": AnthropicModel,
    "meta-llama/Llama-2-70b-chat-hf": OpenModelVLLM,
    "meta-llama/Llama-2-13b-chat-hf": OpenModelVLLM,
    "meta-llama/Llama-2-7b-chat-hf": OpenModelVLLM,
    "meta-llama/Meta-Llama-3-70B-Instruct": OpenModelHF,
    "meta-llama/Llama-3.1-8B-Instruct": OpenModelHF,
    "allenai/tulu-2-7b": OpenModelVLLM,
    "allenai/tulu-2-13b": OpenModelVLLM,
    "allenai/tulu-2-70b": OpenModelVLLM,
    "Salesforce/codet5p-6b": OpenModelHF,
    "Salesforce/codet5p-16b": OpenModelHF,
    "Salesforce/instructcodet5p-16b": OpenModelHF,
    "allenai/unifiedqa-t5-11b": OpenModelHF,
    "Xwin-LM/XwinCoder-7B": OpenModelVLLM,
    "Xwin-LM/XwinCoder-13B": OpenModelVLLM,
    "Xwin-LM/XwinCoder-34B": OpenModelVLLM,
    "Xwin-LM/Xwin-LM-70B-V0.1": OpenModelVLLM,
    "codellama/CodeLlama-34b-Instruct-hf": OpenModelHF, #  Code completion. / Instruction and chat
    "codellama/CodeLlama-34b-Python-hf": OpenModelHF,   #  Code completion. / Python specialist
    "codellama/CodeLlama-34b-hf": OpenModelHF,          #  Code completion
    "bigcode/starcoder": OpenModelVLLM,                 #  StarCoderbase trained on additional 30B tokens of Python
    "bigcode/starcoderplus": OpenModelVLLM,               
    "WizardLM/WizardCoder-Python-34B-V1.0": OpenModelVLLM,
    "google/codegemma-7b-it": OpenModelHF,
    "mistralai/Mistral-7B-v0.1": OpenModelVLLM
}

CODEFORCE_GENERATOR = '''You are a Python code generator, only return the import and python function. Input will be an very detailed description of task, output will be the code.
The input will be from command line, and the output will be printed to the console as well. Your result will be solely a function named solve(), and do not call this function in your code.
Make sure the code is free of bug and can pass the test cases provided. You can use any library you want. The test cases are provided in the code. Do not call the solve() function in your code.'''
CODEX_GENERATOR = '''You are a Python code generator, following the instruction to complete the given code. Only return the import and python function.'''

__PROMPT__ = {
    "codeforce": CODEFORCE_GENERATOR,
    "codex": CODEX_GENERATOR
}

CODE_REVIEWER = '''You are a code reviewer. Detect all the programming techniques from the input and return a list of programming techniques. Only select the techniques from this list: ''' + \
f'{TECHNIQUES}' + \
'''\nYour output should look like this:\n- technique 1\n- technique 2\n- technique 3\n- ...'''

CACHE_DIR="/home/FYP/mohor001/NeoCoder/scratch4"

def get_dp_generate_params(
    dataset_dir: Text,
    model_name: Text,
    num_sample: int,
    dp_rounds: int
):
    """Get the parameters for single thread DP generation. 
    """
    if "codeforce" in dataset_dir.lower():
        dataset = CodeforceDPGenerateDataset(
            dataset_dir,
            num=num_sample,
        )
        dataset_name = "codeforce"
    elif "codex" in dataset_dir.lower():
        dataset = CodexDPGenerateDataset(
            dataset_dir,
            num=num_sample,
        )
        dataset_name = "codex"
    else:
        raise ValueError(f"Dataset {dataset_dir} is not supported.")
    
    try:
        model_class = __MODEL_TO_CLASS__[model_name]
    except KeyError:
        raise ValueError(f"Model {model_name} is not supported.")
    
    assert model_class == OpenAIModel, "Only OpenAIModel is used for generating DP benchmark dataset"
    
    code_generator = model_class(model=model_name, 
                                 gpt_setting=__PROMPT__[dataset_name])
        
    code_reviewer = model_class(model=model_name, 
                                gpt_setting=CODE_REVIEWER)
    
    collate_fn = DPGenerateCollateFn()

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    generator = APIModelSingleThreadDPGenerator(
        code_generator=code_generator,
        code_reviewer=code_reviewer,
        dp_rounds=dp_rounds
    )

    return {
        "generator": generator,
        "dataloader": dataloader,
    }

def get_dp_inference_params(
    dataset_dir: Text,
    model_name: Text,
    dp_rounds: int,
    batch_size: int
):
    """Get the parameters for parallel thread DP inference. 
    """

    if "codeforce" in dataset_dir.lower():
        dataset_name = "codeforce"
    elif "codex" in dataset_dir.lower():
        dataset_name = "codex"
    else:
        raise ValueError(f"Dataset {dataset_dir} is not supported.")

    dataset = DPInferenceDataset(dataset_dir,
                                 dp_rounds=dp_rounds)

    try:
        model_class = __MODEL_TO_CLASS__[model_name]
    except KeyError:
        raise ValueError(f"Model {model_name} is not supported.")

    is_open_model = model_class != OpenAIModel and model_class != AnthropicModel
    use_vllm = model_class == OpenModelVLLM

    if is_open_model:
        model: OpenModel = model_class(model_name=model_name,
                                       prompt=__PROMPT__[dataset_name])
        
        collate_fn = CodeforceDPInferenceCollateFn(
            tokenizer=model.tokenizer,
            is_open_model=is_open_model,
            use_vllm=use_vllm,
            dp_rounds=dp_rounds,
            prompt=model.prompt
        )

    else:
        model: Union[OpenAIModel, AnthropicModel] =  model_class(model=model_name, 
                                                                 gpt_setting=__PROMPT__[dataset_name])
        
        collate_fn = CodeforceDPInferenceCollateFn(
            tokenizer=None,
            is_open_model=False,
            use_vllm=False,
            dp_rounds=dp_rounds,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size if is_open_model else 1,
        sampler=None,
        shuffle=False,
        collate_fn=collate_fn,
    )

    open_model_config = {
        "do_sample": True,
        "max_new_tokens": 1024,
        "num_return_sequences": 1,
        "num_beam_groups": 1,
        "num_beams": 1,
        "temperature": 0.7,   
    } if not use_vllm else {
        "max_tokens": 1024,
        'n': 1,
        'temperature': 0.7,
        "use_beam_search": False,
    }

    generator = OpenModelParallelThreadDPInference(
        model=model,
        dp_rounds=dp_rounds,
        config=open_model_config,
        use_vllm=use_vllm
    ) if is_open_model else \
                APIModelParallelThreadDPInference(
        model=model,
        dp_rounds=dp_rounds,
    ) 

    return {
        "generator": generator,
        "dataloader": dataloader,
    }

def correctness_evaluation(inference_result_path: str,
                           test_case_path: str,
                           save_folder: str,
                           type: str):
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    if "codeforce" in inference_result_path.lower():
        codeforce_correctness_evaluator = CodeForceCorrectnessEvaluator(inference_result_path, 
                                                              test_case_path)
        correctness = codeforce_correctness_evaluator.evaluate()

        model_name = codeforce_correctness_evaluator.model_name
        num_sample = codeforce_correctness_evaluator.num_sample
        num_dp = codeforce_correctness_evaluator.num_dp
        out_file = os.path.join(save_folder, 
                             f"{model_name}_sample={num_sample}_dp={num_dp}_{type}_creativity.json")
        
        with open(out_file, "w") as f:
            json.dump(correctness, f, indent=4)

    elif "codex" in test_case_path.lower():
        codex_correctness_evaluator = CodexCorrectnessEvaluator(inference_result_path,
                                                               test_case_path)
        pass_at_k, results = codex_correctness_evaluator.evaluate()
        print(pass_at_k)

        model_name = codex_correctness_evaluator.model_name
        num_sample = codex_correctness_evaluator.num_sample
        num_dp = codex_correctness_evaluator.num_dp

        out_file = os.path.join(save_folder, 
                             f"{model_name}_sample={num_sample}_dp={num_dp}_creativity.json")
        
        write_json(combine_results(inference_result_path, results), out_file) 
    else:
        raise ValueError("Unknown dataset")

def technique_detection(human_solution_path: str,
                        inference_result_path: str):
    """Detect the programming techniques in human solutions and the generated codes.
    Only applicable to Codeforce dataset
    """

    with open(inference_result_path, "r") as f:
        inference_result = json.load(f)
    with open(human_solution_path, "r") as f:
        human = json.load(f)
    
    save_path = os.path.join(os.path.dirname(human_solution_path), "human_solution_techniques.json")
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            human_tech = json.load(f)
    else:
        human_tech = {}

    code_reviewer = OpenAIModel(model="gpt-5-mini", gpt_setting=CODE_REVIEWER)
    code_evaluator = CodeForceCorrectnessEvaluator(inference_result_path, test_case_path=None) # only used for parsing codes
    code_generator = APIModelParallelThreadDPInference("gpt-5-mini", dp_rounds=0) # only used for parsing techniques
    
    def _detect_techniques(code: Text) -> List[Text]:
        while True:
            technique_str = code_reviewer(code)[0]
            code_reviewer.restart()
            techniques = code_generator.parse_techniques(technique_str)
            # sometimes GPT-4 just repeats all techniques in the prompt
            # in such case we let the model to re-generate the techniques
            if len(techniques) <= 10:
                break

        return techniques
    
    def save_json(data: Dict[str, Any], save_path: str):
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

    for problem in tqdm(inference_result, desc="Detecting techniques"):
        problem_id = problem["problem_id"]
        human_solutions: List[Text] = human[problem_id]
        if "codes" in problem:
            model_generated_codes = []
            for idx, output in enumerate(problem['codes']):
                if output is not None:
                    model_generated_codes.append(code_evaluator.parse_code(output))
                else:
                    # sometimes the OpenAI API returns code not in the ```python ... ``` format
                    model_generated_codes.append(code_evaluator.parse_code(problem['outputs'][idx]))
        else:
            model_generated_codes: List[Text] = [code_evaluator.parse_code(code) for code in problem['outputs']]

        # Detect techniques in human solutions
        if problem_id in human_tech and len(human_tech[problem_id]) == len(human_solutions):
            pass
        else:
            techniques = []
            for human_solution in human_solutions:
                techniques.append(_detect_techniques(human_solution)) if human_solution else techniques.append([])
            
            human_tech.update({problem_id: techniques})
            save_json(human_tech, save_path)

        # Detect techniques in model generated codes
        if "techniques" in problem and len(problem["techniques"]) == len(model_generated_codes) and all(len(tech) <= 10 for tech in problem["techniques"]):
            pass
        else:
            techniques = []
            for model_generated_code in model_generated_codes:
                techniques.append(_detect_techniques(model_generated_code)) if model_generated_code else techniques.append([])
            
            problem.update({"techniques": techniques})
            save_json(inference_result, inference_result_path)

        # monitor OpenAI API usage
        print(code_reviewer.gpt_usage(model = "gpt-5-mini"))

def calculate_creativity(inference_result_path: str,
                         human_solution_path: str,
                         save_folder: str,
                         type: str,
                         dp_rounds: int = 5):
    """Calculate the final creativity score and update the original JSON file
    """
    with open(inference_result_path, "r") as f:
        model_solutions = json.load(f)

    with open(human_solution_path, "r") as f:
        human = json.load(f)
    
    assert "correctness" in model_solutions[-1], "Please run correctness evaluation before calculating creativity."
    assert "techniques" in model_solutions[-1], "Please run technique detection before calculating creativity."

    human_solutions = {k: [t for ts in v for t in ts] for k, v in human.items()}
    human_solutions_counter = {k: Counter(v) for k, v in human_solutions.items()}
    # get the size of each human solutions counter
    human_solutions_size = {k: len(v.values()) for k, v in human_solutions_counter.items()}
    # sort the human_solution_size
    human_solutions_size = dict(sorted(human_solutions_size.items(), key=lambda x: x[1], reverse=True))

    results = dict(problem_id = [], 
                   dp = [],
                   constraints = [], 
                   machine_techniques = [], 
                   human_techniques = [], 
                   machine_solutions = [], 
                   correctness = [])
    
    code_evaluator = CodeForceCorrectnessEvaluator(inference_result_path, test_case_path=None) # only used for parsing codes
    
    for problem in model_solutions:
        problem_id = problem["problem_id"]
        if "codes" in problem:
            model_codes = []
            for idx, output in enumerate(problem['codes']):
                if output is not None: 
                    model_codes.append(code_evaluator.parse_code(output))
                else:
                    # sometimes the OpenAI API returns code not in the ```python ... ``` format
                    model_codes.append(code_evaluator.parse_code(problem['outputs'][idx]))
        elif "outputs" in problem:
            model_codes = [code_evaluator.parse_code(output) for output in problem["outputs"]]
        else:
            raise ValueError("No codes or outputs found in model solutions")

        if "constraints" in problem:
            constraints = problem["constraints"]
        elif "constraints_list" in problem:
            constraints = problem["constraints_list"]
        else:
            raise ValueError("No constraints found in model solutions")

        human_technique = list(human_solutions_counter[problem_id].keys())
        dp_idx = 0
        prev_constraint = None
        for constraint, model_technique, model_code, correctness in zip(constraints, problem["techniques"], model_codes, problem['correctness']):

            if constraint == prev_constraint:
                continue
            else:
                results["problem_id"].append(problem_id)
                results["dp"].append(dp_idx)
                results["constraints"].append(constraint)
                results["machine_techniques"].append(model_technique)
                results["human_techniques"].append(human_technique)
                results["machine_solutions"].append(model_code)
                results["correctness"].append(correctness)
                dp_idx += 1
                prev_constraint = constraint

    results = pd.DataFrame(results)
    results.set_index('problem_id', inplace=True)

    def check_constraints(row):
        return not bool(set(row["machine_techniques"]) & set(row["constraints"]))
    
    def check_techniques(row):
        if row["machine_techniques"] == []:
            return 0
        else:
            return len(set(row["machine_techniques"]) - set(row["human_techniques"]))

    def calcualte_new_techniques_ratio(row):
        if row["machine_techniques"] == []:
            return 0
        else:
            return row['new_techniques'] / len(row['machine_techniques'])

    results["follow_constraints"] = results.apply(check_constraints, axis=1)
    results["new_techniques"] = results.apply(check_techniques, axis=1)
    results["new_techniques_ratio"] = results.apply(calcualte_new_techniques_ratio, axis=1)

    # delete rows 1773F, as we cannot crawl its human solutions due to the website's restriction
    results = results[results.index != "1773F"]

    def calculate_ins_follow(results, dp_rounds):
        """probability of following constraints at dp_rounds
        """

        num_samples = len(results[results["dp"] == dp_rounds])
        num_ins_follow=len(results[(results["dp"] == dp_rounds) & (results["follow_constraints"] == True)])

        
        return 1-(num_ins_follow/num_samples)
    
    def calculate_fac_follow(results, dp_rounds):
        """probability of correctness at dp_rounds
        """

        num_samples = len(results[results["dp"] == dp_rounds])
        
        num_fac_follow=len(results[(results["dp"] == dp_rounds) & (results["correctness"] == True)])
        
        
        return 1-(num_fac_follow/num_samples)

    def calculate_hal_conf(results, dp_rounds):
        """probability of correctness at dp_rounds
        """

        num_samples = len(results[results["dp"] == dp_rounds])
        
        num_fac_ins=len(results[(results["dp"] == dp_rounds) & (results["correctness"] != True) & (results["follow_constraints"] != True)])
        num_fac_noins=len(results[(results["dp"] == dp_rounds) & (results["correctness"] != True) & (results["follow_constraints"] == True)])
        num_nofac_ins=len(results[(results["dp"] == dp_rounds) & (results["correctness"] == True) & (results["follow_constraints"] != True)])
        num_nofac_noins=len(results[(results["dp"] == dp_rounds) & (results["correctness"] == True) & (results["follow_constraints"] == True)])

        return num_fac_ins, num_fac_noins, num_nofac_ins, num_nofac_noins

    def calculate_convergent_thinking(results, dp_rounds):
        """probability of following constraints and correctness at dp_rounds
        """
        num_samples = len(results[results["dp"] == dp_rounds])
        num_correct_samples = len(results[(results["dp"] == dp_rounds) & (results["follow_constraints"] == True) & (results["correctness"] == True)])
        print("dp_rounds is", dp_rounds)
        print("num_samples is", num_samples)
        print("num_correct_samples_is", num_correct_samples)
        
        return num_correct_samples / num_samples

    def calculaate_divergent_thinking(results, dp_rounds):
        """Average number of new techniques at dp_rounds
        """
        return results[results["dp"] == dp_rounds]["new_techniques_ratio"].mean()
    
    def calculate_creativity(results, dp_rounds):
        """Probability of convergent thinking and divergent thinking at dp_rounds
        """
        dp_cluster = results[results["dp"] == dp_rounds]
        num_creative_samples = dp_cluster.apply(lambda x: x["follow_constraints"] * x["correctness"] * x["new_techniques_ratio"], axis=1).sum()
        return num_creative_samples / len(dp_cluster)

    ins_follow=[calculate_ins_follow(results, i) for i in range(0, dp_rounds+1)]
    fac_follow=[calculate_fac_follow(results, i) for i in range(0, dp_rounds+1)]
    convergent_thinking = [calculate_convergent_thinking(results, i) for i in range(0, dp_rounds+1)]
    divergent_thinking = [calculaate_divergent_thinking(results, i) for i in range(0, dp_rounds+1)]
    creativity = [calculate_creativity(results, i) for i in range(0, dp_rounds+1)]
    num_fac_ins, num_fac_noins, num_nofac_ins, num_nofac_noins = zip(*[calculate_hal_conf(results, i) for i in range(0, dp_rounds+1)])

    # Load or create summary.json
    summary_path = f'{save_folder}/summary.json'
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
    else:
        summary_data = {}
    
    # Create a unique key for this run based on the input file and type
    base_name = os.path.basename(inference_result_path).split("_sample")[0]
    run_key = f"{base_name}_{type}"
    cleaned_type = re.sub(r'\d', '', type)
    
    # Store only the aggregate creativity scores for this run
    summary_data[run_key] = {
        "type": cleaned_type,
        "convergent_creativity": convergent_thinking,
        "divergent_creativity": divergent_thinking,
        "total_creativity": creativity,
        "instruction_hallucination": ins_follow,
        "factual_hallucination": fac_follow,
        "num_fac_ins": num_fac_ins,
        "num_fac_noins": num_fac_noins,
        "num_nofac_ins": num_nofac_ins,
        "num_nofac_noins": num_nofac_noins,
        "dp_rounds": dp_rounds
    }
    
    # Save the updated summary.json
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary updated and saved to: {summary_path}")

    # Save CSV results for further analysis
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    base_name = os.path.basename(inference_result_path).split("_sample")[0] 
    results.to_csv(os.path.join(save_folder, base_name + f"_{type}_creativity.csv"))

    return ins_follow,fac_follow,convergent_thinking, divergent_thinking, creativity, num_fac_ins, num_fac_noins, num_nofac_ins, num_nofac_noins