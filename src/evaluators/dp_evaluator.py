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
import tempfile
import shutil
import re
from datetime import datetime

class SandboxErrorLogger:
    """Logs and categorizes errors from sandboxed code execution"""
    
    ERROR_PATTERNS = {
        'timeout': [r'timeout', r'timed out', r'TimeoutError', r'TimeoutException'],
        'memory_error': [r'MemoryError', r'memory', r'out of memory'],
        'recursion_error': [r'RecursionError', r'maximum recursion depth exceeded'],
        'index_error': [
            r'IndexError',
            r'index out of (range|bounds)',
            r'list index out of range',
            r'string index out of range',
            r'tuple index out of range'
        ],
        'division_error': [r'ZeroDivisionError', r'division by zero'],
        'type_error': [r'TypeError', r'unsupported operand type', r'must be .*, not'],
        'value_error': [r'ValueError', r'invalid literal', r'could not convert string to'],
        'attribute_error': [r'AttributeError', r'has no attribute'],
        'import_error': [r'ImportError', r'ModuleNotFoundError', r'No module named'],
        'syntax_error': [r'SyntaxError', r'invalid syntax'],
        'segmentation_fault': [r'Segmentation fault', r'SIGSEGV', r'crashhh'],
        'process_killed': [r'SIGKILL', r'killed', r'terminated']
    }
    
    def __init__(self, log_file: str = "sandbox_errors.json"):
        self.log_file = log_file
        self.errors = []  # List to store ALL errors
        self.error_count = 0
        
    def categorize_error(self, error_message: str, exit_code: Optional[int] = None) -> str:
        """Categorize error based on message patterns"""
        
        if exit_code == -9:
            return "process_killed"
        elif exit_code == -11:
            return "segmentation_fault"
        elif exit_code == -15:
            return "process_terminated"
        
        error_message_lower = error_message.lower()
        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_message_lower, re.IGNORECASE):
                    return error_type
        return "other_error"
    
    def should_log_error(self, error_message: str) -> bool:
        """Determine if error should be logged (exclude input format errors)"""
        
        exclude_patterns = [
            r'not enough values to unpack',
            r'too many values to unpack',
            r'EOF when reading a line',
            r'input\(\) takes',
            r'unexpected EOF while parsing'
        ]
        
        error_lower = error_message.lower()
        for pattern in exclude_patterns:
            if re.search(pattern, error_lower, re.IGNORECASE):
                return False
        return True
    
    def log_error(self, problem_id: str, error_message: str, code_snippet: str, 
                  exit_code: Optional[int] = None, test_mode: str = "unknown"):
        """Log a sandbox execution error - logs ALL errors, not just first per problem"""
        
        if not self.should_log_error(error_message):
            return
        
        error_type = self.categorize_error(error_message, exit_code)
        
        # Add error to list (no deduplication)
        self.error_count += 1
        self.errors.append({
            'error_id': self.error_count,
            'error_type': error_type,
            'problem_id': problem_id,
            'code_snippet': self._truncate_code(code_snippet),
            'error_message': error_message[:200],
            'test_mode': test_mode  # 'batch' or 'individual'
        })
    
    def _truncate_code(self, code: str, max_lines: int = 50) -> str:
        """Truncate code snippet to reasonable size"""
        if not code:
            return ""
        lines = code.split('\n')
        if len(lines) > max_lines:
            return '\n'.join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
        return code
    
    def save_to_file(self):
        """Save error log to JSON file"""
        # Calculate summary statistics
        error_counts = defaultdict(int)
        problem_error_counts = defaultdict(int)
        
        for error in self.errors:
            error_counts[error['error_type']] += 1
            problem_error_counts[error['problem_id']] += 1
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_errors': len(self.errors),
            'unique_problems_with_errors': len(problem_error_counts),
            'error_counts': dict(error_counts),
            'errors_per_problem': dict(problem_error_counts),
            'errors': self.errors
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(output, f, indent=2)
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts by type"""
        error_counts = defaultdict(int)
        for error in self.errors:
            error_counts[error['error_type']] += 1
        return dict(error_counts)

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

# def solver(queue, test_input, module_name="solve_module"):
#     spec = importlib.util.find_spec(module_name)
#     if spec is None:
#         raise ImportError(f"Module {module_name} could not be found.")
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     solve_fn = module.solve
#     with mock_input(test_input):
#         with capture_output() as out:
#             result = solve_fn()
#             printed_output = out.getvalue().strip()
#             if not printed_output and result is not None:
#                 if isinstance(result, list):
#                     printed_output = '\n'.join(str(item) for item in result)
#                 else:
#                     printed_output = str(result)
#             queue.put(printed_output.split('\n') if printed_output else [])

def solver(queue, test_input, module_name="solve_module"):
    """Modified solver that captures and returns error information"""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            queue.put(('error', 'ImportError', f"Module {module_name} could not be found.", None))
            return
            
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
                queue.put(('success', printed_output.split('\n') if printed_output else []))
                
    except Exception as e:
        # Capture the error type and message
        import traceback
        error_type = type(e).__name__
        error_msg = str(e)
        tb = traceback.format_exc()
        queue.put(('error', error_type, error_msg, tb))
                  

logger = logging.getLogger(__name__)


class CodeForceCorrectnessEvaluator(Evaluator):

    def __init__(self,
                 inference_result_path: str,
                 test_case_path: str) -> None:
        super().__init__(inference_result_path, test_case_path)

    @overrides
    def evaluate(self) -> None:
        logger.info(f"Functional Correctness Evaluation: model_name={self.model_name}, \
                    num_sample={self.num_sample}, \
                    num_dp={self.num_dp}")

        # inference_folder = self.inference_result_path[:-5]
        inference_folder= "/".join(self.inference_result_path.split("/")[:-1] + ["errors", self.inference_result_path.split("/")[-1].replace(".json", "_errors.json")])

        # error_logger = SandboxErrorLogger(f"{inference_folder}_errors.json")
        error_logger = SandboxErrorLogger(inference_folder)
        
        inference_result = read_json(self.inference_result_path)
        test_cases = self.read_test_case(self.test_case_path)
        
        for rid, result in enumerate(inference_result):
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

            correctness_all_dps = []
            output_all_dps = []

            for dp_idx, code in enumerate(codes):
                if code is not None:
                    assert len(test_case['input']) == len(test_case['output'])
                    correctness = False
                    output = None

                    try:
                        self._last_error = None
                        self._current_test_mode = 'unknown'
                        
                        (correctness, output) = function_with_timeout(self.test_correctness, (code, test_case['input'], test_case['output']), timeout=6)
                        
                        # Check if an error was recorded during execution
                        if hasattr(self, '_last_error') and self._last_error:
                            error_msg, exit_code = self._last_error
                            test_mode = getattr(self, '_current_test_mode', 'unknown')
                            error_logger.log_error(result['problem_id'], error_msg, code, exit_code, test_mode)
                        
                    except Exception as e:
                        # Log errors that occur
                        if hasattr(self, '_last_error') and self._last_error:
                            error_msg, exit_code = self._last_error
                            test_mode = getattr(self, '_current_test_mode', 'unknown')
                            error_logger.log_error(result['problem_id'], error_msg, code, exit_code, test_mode)
                        else:
                            error_logger.log_error(result['problem_id'], f"Outer execution timeout: {str(e)}", code, None, 'outer_timeout')
                            correctness_all_dps.append(False)
                            output_all_dps.append("code execution timeout")
                            continue
                    finally:
                        if 'solve' in globals():
                            del globals()['solve']

                    if output is None:
                        error_logger.log_error(result['problem_id'], "Code not executable - failed to import or parse", code, None, 'parse_error')
                        correctness_all_dps.append(correctness)
                        output_all_dps.append("code not executable")
                    else:
                        correctness_all_dps.append(correctness)
                        output_all_dps.append(output)
                else:
                    correctness_all_dps.append(False)
                    output_all_dps.append("code not parsable")

            result['correctness'] = correctness_all_dps
            result['output'] = output_all_dps
            result['expected_output'] = test_case['output']

        # Save error log and print summary
        error_logger.save_to_file()
        print("\nError Summary:")
        for error_type, count in error_logger.get_error_summary().items():
            print(f"  {error_type}: {count}")
        
        return inference_result

    # @overrides
    # def evaluate(self) -> None:
    #     logger.info(f"Functional Correctness Evaluation: model_name={self.model_name}, \
    #                   num_sample={self.num_sample}, \
    #                   num_dp={self.num_dp}")

    #     inference_folder = self.inference_result_path.rsplit('/', 1)[0]
    #     error_logger = SandboxErrorLogger(f"{inference_folder}/sandbox_errors.json")
    #     inference_result = read_json(self.inference_result_path)
    #     test_cases = self.read_test_case(self.test_case_path)
        
    #     for rid,result in enumerate(inference_result):
            
    #         test_case = test_cases[result['problem_id']]
    #         if "codes" in result:
    #             codes = []
    #             for idx, output in enumerate(result['codes']):
    #                 if output is not None:
    #                     codes.append(self.parse_code(output))
    #                 else:
    #                     codes.append(self.parse_code(result['outputs'][idx]))
    #         elif "outputs" in result:
    #             codes = [self.parse_code(output) for output in result['outputs']]
    #         else:
    #             raise ValueError("No expected output found in inference result.")

    #         correctness_all_dps = []
    #         output_all_dps = []

    #         for dp_idx, code in enumerate(codes):
          
    #             if code is not None:
    #                 assert len(test_case['input']) == len(test_case['output'])

    #                 try:
    #                     self._last_error = None
    #                     (correctness, output) = function_with_timeout(self.test_correctness, (code, test_case['input'], test_case['output']), timeout=6)
    #                     if hasattr(self, '_last_error') and self._last_error:
    #                         error_msg, exit_code = self._last_error
    #                         error_logger.log_error(result['problem_id'], error_msg, code, exit_code)
                    
    #                 except Exception as e:
    #                     if hasattr(self, '_last_error') and self._last_error:
    #                         error_msg, exit_code = self._last_error
    #                         error_logger.log_error(result['problem_id'], error_msg, code, exit_code)
    #                     else:
    #                         error_logger.log_error(result['problem_id'], str(e), code)
    #                     # correctness: False, output: "code execution timeout"
    #                     correctness_all_dps.append(False)
    #                     output_all_dps.append("code execution timeout")
    #                     continue
    #                 finally:
    #                     # remove solve() function if exists
    #                     if 'solve' in globals():
    #                         del globals()['solve']

    #                 if output is None:
    #                     # correctness: False, output: "code not executable"
    #                     correctness_all_dps.append(correctness)
    #                     output_all_dps.append("code not executable")
    #                 else:
    #                     # correctness: bool, output: List[Text]
    #                     correctness_all_dps.append(correctness)
    #                     output_all_dps.append(output)
    #             else:
    #                 # correctness: False, output: "code not parsable"
    #                 correctness_all_dps.append(False)
    #                 output_all_dps.append("code not parsable")

    #         result['correctness'] = correctness_all_dps
    #         result['output'] = output_all_dps
    #         result['expected_output'] = test_case['output']
        
    #     error_logger.save_to_file()
    #     print("\nError Summary:")
    #     for error_type, count in error_logger.get_error_summary().items():
    #         print(f"  {error_type}: {count}")

    #     return inference_result

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

    def execute_solve(self, test_input, module_name: str = "solve_module"):
        module = sys.modules[module_name]
        
        # Queue to capture results from the subprocess
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=solver, args=(queue, test_input,))
        process.start()
        process.join(timeout=60)

        # Handle timeout
        if process.is_alive():
            process.terminate()
            process.join()
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            
            # Store timeout error
            self._last_error = ('Process execution timeout (60s exceeded)', -9)
            raise Exception('timeout!')

        # Try to get result from queue
        try:
            result = queue.get_nowait()
            
            # Check if it's an error response
            if isinstance(result, tuple) and len(result) >= 2:
                if result[0] == 'error':
                    # We have error details!
                    _, error_type, error_msg, tb = result
                    self._last_error = (f"{error_type}: {error_msg}", process.exitcode)
                    raise Exception("oops crashhh")
                elif result[0] == 'success':
                    # Success case
                    return result[1]
            
            # Backward compatibility - return as-is
            return result
            
        except:
            # No result in queue or other issue
            if process.exitcode != 0:
                # Process crashed but no details available
                self._last_error = (f"Process terminated abnormally with exit code: {process.exitcode}", process.exitcode)
                raise Exception("oops crashhh")
            else:
                # Some other issue
                raise
    
    def test_correctness(self, code, test_case_inputs, test_case_outputs):
        try:
            temp_dir, file_path = write_solve_to_file(code)
            import_solve_from_file(file_path, temp_dir)
        except Exception as e:
            return False, None

        # Track which mode we're in for error logging
        self._current_test_mode = 'batch'
        
        try:
            # FIRST TRY: Batch execution with all test cases
            num_test_cases = len(test_case_inputs)
            test_input = [" ".join(row) for case in test_case_inputs for row in case]
            test_input.insert(0, str(num_test_cases))
            
            try:
                output = self.execute_solve(test_input)
            except Exception as exec_error:
                if "not enough values to unpack" in str(exec_error):
                    pass  # Will trigger fallback
                else:
                    raise
            
            correctness = []
            for i, (out, test_out) in enumerate(zip(output, test_case_outputs)):
                is_correct = type_agnostic_compare(out, test_out)
                correctness.append(is_correct)
            
            all_correct = all(correctness)
            return all_correct, output
            
        except Exception as e:
            if "not enough values to unpack" not in str(e):
                raise
            pass  # Move to fallback mode

        # Individual test case execution (FALLBACK)
        self._current_test_mode = 'individual'
        output = []
        correctness = []
        
        for i, (test_case_input, test_case_output) in enumerate(zip(test_case_inputs, test_case_outputs)):
            try:
                individual_input = [" ".join(row) for row in test_case_input]
                
                output_ = self.execute_solve(individual_input)
                
                # Smart flattening logic...
                if isinstance(test_case_output, list) and len(test_case_output) > 1:
                    flattened_output = output_
                elif isinstance(test_case_output, str) or (isinstance(test_case_output, list) and len(test_case_output) == 1):
                    if output_ and len(output_) == 1:
                        flattened_output = output_[0]
                    else:
                        flattened_output = output_
                else:
                    flattened_output = output_
                
                is_correct = type_agnostic_compare(flattened_output, test_case_output)
                correctness.append(is_correct)
                output.append(flattened_output)
                
            except Exception as e:
                if "not enough values to unpack" not in str(e):
                    raise
                correctness.append(False)
                output.append(None)
        
        all_correct = all(correctness)
        
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        return all_correct, output

    # def execute_solve(self,test_input,module_name: str = "solve_module"):

    #     module = sys.modules[module_name]

    # # Access the solve function directly from the module
    #     solve_fn = module.solve
            
    #     # Queue to capture results from the subprocess
    #     queue = multiprocessing.Queue()
    #     process = multiprocessing.Process(target=solver,args=(queue,test_input,))
    #     process.start()
    #     process.join()  # No timeout; wait until completion

    #     process.join(timeout=60)

    #     # If the process is still alive after 60s, kill it and return 0
    #     if process.is_alive():
    #         process.terminate()
    #         process.join()  # clean up
    #         if process.is_alive():
    #             process.kill()  # or os.kill(..., SIGKILL)
    #             process.join(timeout=5)
    #         self._last_error = ('timeout!', -9)
    #         raise Exception('timeout!')

    #     if process.exitcode != 0:
    #         # Process terminated abnormally (e.g., SIGKILL)
    #         self._last_error = (f"Process terminated abnormally with exit code: {process.exitcode}", process.exitcode)
    #         raise Exception("oops crashhh")
    #     return queue.get()

    # def test_correctness(self, code, test_case_inputs, test_case_outputs):
    #     try:
    #         temp_dir, file_path = write_solve_to_file(code)
    #         import_solve_from_file(file_path, temp_dir)
    #     except Exception as e:
    #         return False, None

    #     try:
    #         # FIRST TRY: Batch execution with all test cases
    #         num_test_cases = len(test_case_inputs)
    #         test_input = [" ".join(row) for case in test_case_inputs for row in case]
    #         test_input.insert(0, str(num_test_cases))
            
    #         # Enhanced execute_solve with better error capture
    #         try:
    #             output = self.execute_solve(test_input)
    #         except Exception as exec_error:
    #             # Check if it's the "not enough values to unpack" error
    #             if "not enough values to unpack" in str(exec_error):
    #                 pass  # Will trigger fallback
    #             raise  # Re-raise to trigger fallback
            
    #         correctness = []
    #         for i, (out, test_out) in enumerate(zip(output, test_case_outputs)):
    #             is_correct = type_agnostic_compare(out, test_out)
    #             correctness.append(is_correct)
            
    #         all_correct = all(correctness)
    #         return all_correct, output
            
    #     except Exception as e:
    #         if "not enough values to unpack" not in str(e):
    #             raise
    #         pass  # Move to fallback mode

    #     # Individual test case execution (FALLBACK)
    #     output = []
    #     correctness = []
        
    #     for i, (test_case_input, test_case_output) in enumerate(zip(test_case_inputs, test_case_outputs)):
    #         try:
    #             # FIX: Use test_case_input, not test_input
    #             individual_input = [" ".join(row) for row in test_case_input]
                
    #             try:
    #                 output_ = self.execute_solve(individual_input)
    #             except Exception as exec_error:
    #                 raise
                
    #             # SMART FLATTENING: Only flatten if expected output is a single item
    #             if isinstance(test_case_output, list) and len(test_case_output) > 1:
    #                 # Expected output has multiple items - don't flatten
    #                 flattened_output = output_
    #             elif isinstance(test_case_output, str) or (isinstance(test_case_output, list) and len(test_case_output) == 1):
    #                 # Expected output is single item - flatten if needed
    #                 if output_ and len(output_) == 1:
    #                     flattened_output = output_[0]
    #                 else:
    #                     flattened_output = output_
    #             else:
    #                 # Default: don't flatten
    #                 flattened_output = output_
                
    #             is_correct = type_agnostic_compare(flattened_output, test_case_output)
    #             correctness.append(is_correct)
    #             output.append(flattened_output)
                
    #         except Exception as e:
    #             # Code is not executable for this specific test case
    #             if "not enough values to unpack" not in str(e):
    #                 raise
    #             correctness.append(False)
    #             output.append(None)
        
    #     all_correct = all(correctness)
        
    #     # Clean up temporary directory
    #     try:
    #         shutil.rmtree(temp_dir)
    #     except:
    #         pass
        
    #     return all_correct, output

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

