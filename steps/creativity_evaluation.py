import click
import os
from src.utils.configs import correctness_evaluation, technique_detection, calculate_creativity

@click.command()
@click.option("--task", type=click.Choice(["correctness", "detection", "creativity"]), help="Task to perform.")
@click.option("--inference-result-path", type=click.Path(exists=True), help="File Path of the inference result of dp dataset.")
@click.option("--human-solution-path", type=click.Path(exists=True), help="File Path of the human solutions", default=None)
@click.option("--test-case-path", type=click.Path(exists=True), help="File Path of the test case of dp dataset.", default=None)
@click.option("--save-folder", type=click.Path(), help="Folder to save the evaluation result.", default=None)
@click.option("--type", type=str, help="Type of hallucination reduction method.", default=None)

def main(
    task,
    inference_result_path,
    human_solution_path,
    test_case_path,
    save_folder,
    type
):
    if task == "detection":
        assert human_solution_path is not None, "Please provide human solution path."

        technique_detection(human_solution_path=human_solution_path,
                            inference_result_path=inference_result_path)
    elif task == "correctness":
        assert test_case_path is not None, "Please provide test case path."
        correctness_evaluation(inference_result_path=inference_result_path,
                               test_case_path=test_case_path,
                               save_folder=save_folder,
                               type= type)
    elif task == "creativity":
        ins_follow,fac_follow,convergent_thinking, divergent_thinking, creativity,num_fac_ins, num_fac_noins, num_nofac_ins, num_nofac_noins = calculate_creativity(
            human_solution_path=human_solution_path,
            inference_result_path=inference_result_path,
            save_folder=save_folder,
            type=type
            )
        print(f"Instruction-Following Hallucination: {ins_follow}")
        print(f"Factuality Hallucination: {fac_follow}")
        print(f"Convergent Thinking: {convergent_thinking}")
        print(f"Divergent Thinking: {divergent_thinking}")
        print(f"Creativity: {creativity}")
        print(f"FactualityHal-InstructionHal: {num_fac_ins}")
        print(f"FactualityHal-NoInstructionHal: {num_fac_noins}")
        print(f"NoFactualityHal-InstructionHal: {num_nofac_ins}")
        print(f"NoFactualityHal-NoInstructionHal: {num_nofac_noins}")
    else:
        raise ValueError("Invalid task.")


if __name__ == "__main__":
    main()
