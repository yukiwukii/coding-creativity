from tqdm import tqdm
import os
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from overrides import overrides
from typing import Text, Dict, Any, List, Optional, Tuple, Union
from .evaluator import Evaluator
from .evaluation_utils import (estimate_pass_at_k, 
                               check_correctness, 
                               mock_input, 
                               capture_output, 
                               type_agnostic_compare,
                               function_with_timeout,
                               stream_json,
                               read_json)

import json
import logging
import numpy as np
import multiprocessing
import sys
import importlib.util
import os
import tempfile
import sys
import shutil

def write_solve_to_file(code: str) -> str:
    """
    Write the dynamically generated code to a temporary Python file.
    Return the directory and file name.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "solve_module.py")

    # Write the code to the file
    with open(file_path, "w") as f:
        f.write(code)

    return temp_dir, file_path

def import_solve_from_file(file_path: str, temp_dir: str):
    """
    Dynamically import the solve function from the given file.
    Add the directory to sys.path for proper importing.
    """
    module_name = "solve_module"

    # Add the temporary directory to sys.path
    sys.path.insert(0, temp_dir)

    # Import the module dynamically
    if module_name in sys.modules:
        del sys.modules[module_name]  # Ensure a fresh import

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Add the module to sys.modules for consistent referencing
    sys.modules[module_name] = module

    # Return the solve function from the module
    return module.solve



# def solver(queue,test_input, module_name="solve_module"):
#     spec = importlib.util.find_spec(module_name)
#     if spec is None:
#         raise ImportError(f"Module {module_name} could not be found.")
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     solve_fn = module.solve

#     # Mock input and capture output
#     with mock_input(test_input):
#         with capture_output() as out:
#             solve_fn()
#             queue.put(out.getvalue().strip().split('\n'))

def solver(queue, test_input, module_name="solve_module"):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f"Module {module_name} could not be found.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solve_fn = module.solve
    with mock_input(test_input):
        with capture_output() as out:
            result = solve_fn()
            printed_output = out.getvalue().strip()
            if not printed_output and result is not None:
                if isinstance(result, list):
                    printed_output = '\n'.join(str(item) for item in result)
                else:
                    printed_output = str(result)
            queue.put(printed_output.split('\n') if printed_output else [])
                  

logger = logging.getLogger(__name__)


class CodeForceCorrectnessEvaluator(Evaluator):
    """
    """

    def __init__(self,
                 inference_result_path: str,
                 test_case_path: str) -> None:
        super().__init__(inference_result_path, test_case_path)

    @overrides
    def evaluate(self) -> None:
        """
        """
        logger.info(f"Functional Correctness Evaluation: model_name={self.model_name}, \
                      num_sample={self.num_sample}, \
                      num_dp={self.num_dp}")
        
        inference_result = read_json(self.inference_result_path)

        test_cases = self.read_test_case(self.test_case_path)
        
        for rid,result in enumerate(inference_result):
            
            test_case = test_cases[result['problem_id']]
            if "codes" in result:
                codes = []
                for idx, output in enumerate(result['codes']):
                    if output is not None:
                        codes.append(self.parse_code(output))
                    else:
                        codes.append(self.parse_code(result['outputs'][idx]))
            elif "outputs" in result:
                codes = [self.parse_code(output) for output in result['outputs']]
            else:
                raise ValueError("No expected output found in inference result.")

            # if "codes" in result:
            #     codes = []
            #     for idx, output in enumerate(result['codes']):
            #         print(f"\n{'='*50}")
            #         print(f"PROBLEM {result['problem_id']} - DP ROUND {idx}")
            #         print(f"{'='*50}")
                    
            #         if output is not None:
            #             print(f"\n--- ORIGINAL CODE (BEFORE PARSING) ---")
            #             print(repr(output))  # Shows escape characters
            #             print("\n--- ORIGINAL CODE (READABLE) ---")
            #             print(output)
            #             print(f"--- END ORIGINAL CODE ---")
                        
            #             parsed_code = self.parse_code(output)
                        
            #             print(f"\n--- PARSED CODE (AFTER PARSING) ---")
            #             print(repr(parsed_code) if parsed_code else "None")  # Shows escape characters
            #             print("\n--- PARSED CODE (READABLE) ---")
            #             print(parsed_code if parsed_code else "None")
            #             print(f"--- END PARSED CODE ---\n")
                        
            #             codes.append(parsed_code)
            #         else:
            #             print(f"\n--- TRYING FROM OUTPUTS[{idx}] ---")
            #             original_from_outputs = result['outputs'][idx]
            #             print(f"ORIGINAL FROM OUTPUTS:")
            #             print(repr(original_from_outputs))
            #             print(original_from_outputs)
                        
            #             parsed_from_outputs = self.parse_code(original_from_outputs)
            #             print(f"PARSED FROM OUTPUTS:")
            #             print(repr(parsed_from_outputs) if parsed_from_outputs else "None")
            #             print(parsed_from_outputs if parsed_from_outputs else "None")
            #             print(f"--- END ---\n")
                        
            #             codes.append(parsed_from_outputs)
                        
            # elif "outputs" in result:
            #     codes = []
            #     for idx, output in enumerate(result['outputs']):
            #         print(f"\n{'='*50}")
            #         print(f"PROBLEM {result['problem_id']} - DP ROUND {idx}")
            #         print(f"{'='*50}")
                    
            #         print(f"\n--- ORIGINAL OUTPUT (BEFORE PARSING) ---")
            #         print(repr(output))
            #         print("\n--- ORIGINAL OUTPUT (READABLE) ---")
            #         print(output)
            #         print(f"--- END ORIGINAL OUTPUT ---")
                    
            #         parsed_code = self.parse_code(output)
                    
            #         print(f"\n--- PARSED OUTPUT (AFTER PARSING) ---")
            #         print(repr(parsed_code) if parsed_code else "None")
            #         print("\n--- PARSED OUTPUT (READABLE) ---")
            #         print(parsed_code if parsed_code else "None")
            #         print(f"--- END PARSED OUTPUT ---\n")
                    
            #         codes.append(parsed_code)
            # else:
            #     raise ValueError("No expected output found in inference result.")

            correctness_all_dps = []
            output_all_dps = []

            for dp_idx, code in enumerate(codes):
          
                if code is not None:
                    assert len(test_case['input']) == len(test_case['output'])

                    try:
                        (correctness, output) = function_with_timeout(self.test_correctness, (code, test_case['input'], test_case['output']), timeout=6)
                    except Exception as e:
                        
                        # correctness: False, output: "code execution timeout"
                        correctness_all_dps.append(False)
                        output_all_dps.append("code execution timeout")
                        continue
                    finally:
                        # remove solve() function if exists
                        if 'solve' in globals():
                            del globals()['solve']


                    if output is None:
                        # correctness: False, output: "code not executable"
                        correctness_all_dps.append(correctness)
                        output_all_dps.append("code not executable")
                    else:
                        # correctness: bool, output: List[Text]
                        correctness_all_dps.append(correctness)
                        output_all_dps.append(output)
                else:
                    # correctness: False, output: "code not parsable"
                    correctness_all_dps.append(False)
                    output_all_dps.append("code not parsable")

            result['correctness'] = correctness_all_dps
            result['output'] = output_all_dps
            result['expected_output'] = test_case['output']

        return inference_result
            

    # def parse_code(self, code: Text) -> Optional[Text]:
    #     """Parse the response generated by the model to get code.
    #     """

    #     assert code is not None, "Code is None."

    #     code = code.replace("sys.stdin.read()", "input()")
    #     code = code.replace("stdin.read()", "input()")
    #     code = code.replace("sys.stdin.readlines()", "input()")
    #     code = code.replace("stdin.readlines()", "input()")
    #     code = code.replace("sys.stdin.readline()", "input()")
    #     code = code.replace("stdin.readline()", "input()")

    #     # Note that different models may have different response format.
    #     # Find the first "def " signature and look back to the first import statement
    #     # either in the form of "import ..." or "from ... import ...".
    #     # if not found, then return the whole code after "def " signature.
    #     def_idx = code.find("def")
    #     if def_idx == -1:
    #         return None
    #     else: 
    #         # return the lowest index of string
    #         import_idx = code.find("import", 0, def_idx)
    #         from_idx = code.find("from", 0, def_idx)
    #         # no import statement
    #         if import_idx == -1 and from_idx == -1:
    #             code =  code[def_idx:]
    #         else:
    #             # "from ... import ..."
    #             if import_idx > from_idx and from_idx != -1:
    #                 code = code[from_idx:]
    #             # "import ..."
    #             elif import_idx != -1 and from_idx == -1:
    #                 code = code[import_idx:]
    #             # "import ... \nfrom ... import ..."
    #             elif import_idx != -1 and from_idx > import_idx:
    #                 code = code[import_idx:]
    #             else:
    #                 # only has "from" no "import" statement
    #                 return None
        
    #     solve_idx = code.find("solve(")
    #     prefix_code = code[:solve_idx]
    #     suffix_code = code[solve_idx:]
    #     lines = suffix_code.split("\n")
    #     for line in lines:
    #         if line.startswith(" ") or line.startswith("\t") or line.startswith("solve"):
    #             prefix_code += line + "\n"
    #         else:
    #             # solve() function ends
    #             break
    #     code = prefix_code

    #     return code

    def parse_code(self, code: Text) -> Optional[Text]:
        """Parse the response generated by the model to get code.
        Handles blank lines, comments, and proper indentation tracking.
        """
        assert code is not None, "Code is None."
        
        # Replace stdin variations with input()
        code = code.replace("sys.stdin.read()", "input()")
        code = code.replace("stdin.read()", "input()")
        code = code.replace("sys.stdin.readlines()", "input()")
        code = code.replace("stdin.readlines()", "input()")
        code = code.replace("sys.stdin.readline()", "input()")
        code = code.replace("stdin.readline()", "input()")
        
        # Find the first "def " signature
        def_idx = code.find("def")
        if def_idx == -1:
            return None
        
        # Look for import statements before def
        import_idx = code.find("import", 0, def_idx)
        from_idx = code.find("from", 0, def_idx)
        
        # Determine starting point based on imports
        if import_idx == -1 and from_idx == -1:
            # No import statements
            start_idx = def_idx
        elif import_idx == -1 and from_idx != -1:
            # Only "from ... import ..."
            start_idx = from_idx
        elif import_idx != -1 and from_idx == -1:
            # Only "import ..."
            start_idx = import_idx
        elif import_idx != -1 and from_idx != -1:
            # Both exist, use the earlier one
            start_idx = min(import_idx, from_idx)
        else:
            return None
        
        # Extract code starting from imports/def
        code = code[start_idx:]
        
        # Find solve() function specifically
        solve_idx = code.find("solve(")
        if solve_idx == -1:
            return None
        
        prefix_code = code[:solve_idx]  # Everything before "solve("
        suffix_code = code[solve_idx:]  # Everything from "solve(" onwards
        
        lines = suffix_code.split("\n")
        function_started = False
        function_indent_level = None
        
        for i, line in enumerate(lines):
            # First line should contain "solve(" 
            if i == 0:
                prefix_code += line + "\n"
                if line.strip().endswith(":"):  # def solve():
                    function_started = True
                    function_indent_level = len(line) - len(line.lstrip())
                continue
            
            if not function_started:
                # Still looking for the function definition line
                prefix_code += line + "\n"
                if line.strip().endswith(":"):
                    function_started = True
                    function_indent_level = len(line) - len(line.lstrip())
                continue
            
            # We're inside the function now
            
            # Handle empty lines - always include them
            if line.strip() == "":
                prefix_code += line + "\n"
                continue
            
            # Handle comments at any indentation level within the function
            if line.strip().startswith("#"):
                prefix_code += line + "\n"
                continue
            
            # Check indentation for actual code
            current_indent = len(line) - len(line.lstrip())
            
            # If this line is indented more than the function definition, it's part of the function
            if current_indent > function_indent_level:
                prefix_code += line + "\n"
                continue
            
            # If this line is at the same indentation as function def or less,
            # and it's not empty or a comment, the function has ended
            if current_indent <= function_indent_level:
                # Check if it's another function/class definition or main block
                stripped = line.strip()
                if (stripped.startswith("def ") or 
                    stripped.startswith("class ") or 
                    stripped.startswith("if __name__") or
                    stripped.startswith("while ") or
                    stripped.startswith("for ") or
                    stripped.startswith("if ") or
                    (stripped and not stripped.startswith(" ") and not stripped.startswith("\t"))):
                    # Function has ended
                    break
            
            # If we get here, include the line
            prefix_code += line + "\n"
        
        return prefix_code

    
    def read_test_case(self, test_case_path: str):
        with open(test_case_path, "r") as f:
            examples = json.load(f)

        test_cases = {}
        for example in examples:
            problem_id = example['problem_id']
            if problem_id not in test_cases:
                test_cases[problem_id] = {'input': example['input'], 'output': example['output']}
            else:
                raise ValueError(f"Duplicate problem_id: {problem_id}")
        
        return test_cases

    def execute_solve(self,test_input,module_name: str = "solve_module"):

        module = sys.modules[module_name]

    # Access the solve function directly from the module
        solve_fn = module.solve
            
        # Queue to capture results from the subprocess
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=solver,args=(queue,test_input,))
        process.start()
        process.join()  # No timeout; wait until completion

        process.join(timeout=60)

        # If the process is still alive after 60s, kill it and return 0
        if process.is_alive():
            process.terminate()
            process.join()  # clean up
            if process.is_alive():
                process.kill()  # or os.kill(..., SIGKILL)
                process.join(timeout=5)
                raise Exception('timeout!')

        if process.exitcode != 0:
            # Process terminated abnormally (e.g., SIGKILL)
            
            raise Exception("oops crashhh")
        return queue.get()

    # def test_correctness(self, 
    #                      code: Text, 
    #                      test_case_inputs: List[List[List[Text]]], 
    #                      test_case_outputs: List[Union[List[Text], Text]]
    #                      ):
    #     """Test the correctness of the code with the given test case.
    #     Output: 
    #         1. Code is executable: bool, List[Text | None] 
    #         where None indicates the code is not executable for the specific given test case.
    #         2. Code is not executable: False, None
    #     """

    #     try:
           
    #         # exec(code, globals())
    #         temp_dir, file_path = write_solve_to_file(code)

    #     # Import the solve function dynamically and set up the module
    #         import_solve_from_file(file_path, temp_dir)
    #     except:
    #         # code is not executable
           
    #         return False, None
        
    #     try:
    #         # first try: feed testing cases at once
            
    #         num_test_cases = len(test_case_inputs)
    #         test_input = [" ".join(row) for case in test_case_inputs for row in case]
    #         test_input.insert(0, str(num_test_cases))
    #         output=self.execute_solve(test_input)   
    #         correctness = [type_agnostic_compare(out, test_out) for out, test_out in zip(output, test_case_outputs)]
    #         return all(correctness), output             
    #     except:
    #         # second try: feed testing cases one by one
            
    #         output = []
    #         correctness = []
    #         for test_case_input, test_case_output in zip(test_case_inputs, test_case_outputs):

    #             test_input = [" ".join(row) for row in test_case_input]
    #             try:
    #                 output_=self.execute_solve(test_input) 
    #                         # if isinstance(test_case_output, list):
    #                         #     output_ = output_.split('\n')
                    
    #                 sys.__stdout__.write("Output is" + output_ + "\n")

    #                 correctness.append(type_agnostic_compare(output_, test_case_output))
    #                 output.append(output_)
    #             except:
    #                         # code is not executable
    #                 correctness.append(False)
    #                 output.append(None)
    #         return all(correctness), output

    # def test_correctness(self, code, test_case_inputs, test_case_outputs):
    #     """Debug version that shows the generated code"""
        
    #     print(f"\n=== Testing Code ===", flush=True)
    #     print(f"Input test cases: {len(test_case_inputs)}", flush=True)
    #     print(f"Expected outputs: {test_case_outputs}", flush=True)
        
    #     # SHOW THE ACTUAL GENERATED CODE
    #     print(f"\n--- GENERATED CODE ---", flush=True)
    #     print(code, flush=True)
    #     print(f"--- END GENERATED CODE ---\n", flush=True)
        
    #     try:
    #         temp_dir, file_path = write_solve_to_file(code)
    #         import_solve_from_file(file_path, temp_dir)
    #         print("Code compilation successful", flush=True)
    #     except Exception as e:
    #         print(f"Code compilation failed: {e}", flush=True)
    #         return False, None  
    
    #     try:
    #         # FIRST TRY: Batch execution with all test cases
    #         num_test_cases = len(test_case_inputs)
    #         test_input = [" ".join(row) for case in test_case_inputs for row in case]
    #         test_input.insert(0, str(num_test_cases))
            
    #         print(f"Attempting batch execution with input: {test_input}", flush=True)
            
    #         # Force output before potentially crashing subprocess
    #         sys.stdout.flush()
    #         sys.stderr.flush()
            
    #         output = self.execute_solve(test_input)   
            
    #         print(f"Batch execution output: {output}", flush=True)
    #         correctness = [type_agnostic_compare(out, test_out) for out, test_out in zip(output, test_case_outputs)]
    #         print(f"Batch execution correctness: {correctness}", flush=True)
            
    #         return all(correctness), output             
            
    #     except Exception as e:
    #         print(f"Batch execution failed: {e}", flush=True)
    #         print("=== STARTING FALLBACK: Individual test case execution ===", flush=True)
            
    #         # SECOND TRY: Individual test case execution (FALLBACK)
    #         output = []
    #         correctness = []
            
    #         for i, (test_case_input, test_case_output) in enumerate(zip(test_case_inputs, test_case_outputs)):
    #             print(f"\n--- Fallback Test Case {i+1} ---", flush=True)
    #             print(f"Raw test_case_input: {test_case_input}", flush=True)
                
    #             test_input = [" ".join(row) for row in test_case_input]
    #             print(f"Processed test_input for individual execution: {test_input}", flush=True)
    #             print(f"Expected output: {test_case_output}", flush=True)
                
    #             try:
    #                 output_ = self.execute_solve(test_input) 
    #                 print(f"Individual execution raw output: {output_}", flush=True)
                    
    #                 # FIX: Individual execution returns a list, but we want to flatten it
    #                 # for single-output test cases
    #                 if output_ and len(output_) == 1:
    #                     # Single output case: ['YES'] -> 'YES'
    #                     flattened_output = output_[0]
    #                 else:
    #                     # Multiple output case: keep as list
    #                     flattened_output = output_
                        
    #                 print(f"Individual execution flattened output: {flattened_output}", flush=True)
    #                 print(f"Expected output: {test_case_output}", flush=True)
                    
    #                 correctness.append(type_agnostic_compare(flattened_output, test_case_output))
    #                 output.append(flattened_output)  # Append the flattened result
                    
    #                 print(f"Individual execution correct: {correctness[-1]}", flush=True)
                    
    #             except Exception as e:
    #                 print(f"Individual execution failed: {e}", flush=True)
    #                 # Code is not executable for this specific test case
    #                 correctness.append(False)
    #                 output.append(None)
            
    #         print(f"=== FALLBACK COMPLETE ===", flush=True)
    #         print(f"Final correctness: {correctness}", flush=True)
    #         print(f"Final output: {output}", flush=True)
            
    #         return all(correctness), output

    def test_correctness(self, code, test_case_inputs, test_case_outputs):
        try:
            temp_dir, file_path = write_solve_to_file(code)
            import_solve_from_file(file_path, temp_dir)
        except Exception as e:
            return False, None

        try:
            # FIRST TRY: Batch execution with all test cases
            num_test_cases = len(test_case_inputs)
            test_input = [" ".join(row) for case in test_case_inputs for row in case]
            test_input.insert(0, str(num_test_cases))
            
            # Enhanced execute_solve with better error capture
            try:
                output = self.execute_solve(test_input)
            except Exception as exec_error:
                # Check if it's the "not enough values to unpack" error
                if "not enough values to unpack" in str(exec_error):
                    pass  # Will trigger fallback
                raise  # Re-raise to trigger fallback
            
            correctness = []
            for i, (out, test_out) in enumerate(zip(output, test_case_outputs)):
                is_correct = type_agnostic_compare(out, test_out)
                correctness.append(is_correct)
            
            all_correct = all(correctness)
            return all_correct, output
            
        except Exception as e:
            pass  # Move to fallback mode

        # Individual test case execution (FALLBACK)
        output = []
        correctness = []
        
        for i, (test_case_input, test_case_output) in enumerate(zip(test_case_inputs, test_case_outputs)):
            try:
                # FIX: Use test_case_input, not test_input
                individual_input = [" ".join(row) for row in test_case_input]
                
                try:
                    output_ = self.execute_solve(individual_input)
                except Exception as exec_error:
                    raise
                
                # SMART FLATTENING: Only flatten if expected output is a single item
                if isinstance(test_case_output, list) and len(test_case_output) > 1:
                    # Expected output has multiple items - don't flatten
                    flattened_output = output_
                elif isinstance(test_case_output, str) or (isinstance(test_case_output, list) and len(test_case_output) == 1):
                    # Expected output is single item - flatten if needed
                    if output_ and len(output_) == 1:
                        flattened_output = output_[0]
                    else:
                        flattened_output = output_
                else:
                    # Default: don't flatten
                    flattened_output = output_
                
                is_correct = type_agnostic_compare(flattened_output, test_case_output)
                correctness.append(is_correct)
                output.append(flattened_output)
                
            except Exception as e:
                # Code is not executable for this specific test case
                correctness.append(False)
                output.append(None)
        
        all_correct = all(correctness)
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        return all_correct, output

class CodexCorrectnessEvaluator(Evaluator):
    def __init__(self,
                 inference_result_path: str,
                 test_case_path: str) -> None:
        super().__init__(inference_result_path, test_case_path)
        logger.info(f"Functional Correctness Evaluation: model_name={self.model_name}, \
                      num_sample={self.num_sample}, \
                      num_dp={self.num_dp}")

    @overrides
    def evaluate(
        self,
        k: List[int] = [1, 10, 100],
        n_workers: int = 4,
        timeout: float = 3.0
    ):
        """
        Evaluates the functional correctness of generated samples, and writes
        results to f"{sample_file}_results.jsonl"
        """
        problems = read_json(self.test_case_path)
        sample_file = self.inference_result_path

        # Check the generated samples against test suites.
        with ThreadPoolExecutor(max_workers=n_workers) as executor:

            futures = []
            completion_id = Counter()
            n_samples = 0
            results = defaultdict(list)

            print("Reading samples...")
            for sample in tqdm(stream_json(sample_file)):
                task_id = sample["problem_id"]
                completions = sample["codes"] if "codes" in sample else sample["outputs"]
                for completion in completions:
                    args = (problems[task_id], completion, timeout, completion_id[task_id])
                    future = executor.submit(check_correctness, *args)
                    futures.append(future)
                    completion_id[task_id] += 1
                    n_samples += 1

            assert len(completion_id) == len(problems), "Some problems are not attempted."

            print("Running test suites...")
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))

        # Calculate pass@k.
        total, correct = [], []
        for result in results.values():
            result.sort()
            passed = [r[1]["passed"] for r in result]
            for p in passed:
                # each dp should be considered as an independent sample
                total.append(1)
                correct.append(1) if p else correct.append(0)

        total = np.array(total)
        correct = np.array(correct)

        ks = k
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                     for k in ks if (total >= k).all()}

        return pass_at_k, results

