from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Decide on a token limit for thinking; As the model's max tokens is 32768, 32000 usually ensures there is enough space for the model to still answer
MAX_TOKENS_THINKING = 800
# Decide how often to ignore end-of-thinking token
NUM_IGNORE = 1

model = LLM(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # s1 originally gets this prompt wrong but with budget forcing it fixes it
    tensor_parallel_size=2,
)
tok = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

stop_token_ids = tok("<|im_end|>")["input_ids"]
sampling_params = SamplingParams(
    max_tokens=32768,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=False,
    temperature=0.0,
)

# For the exact raspberry sample in the paper see
prompts = [
'''You are a Python code generator, only return the import and python function. Input will be an very detailed description of task, output will be the code.
The input will be from command line, and the output will be printed to the console as well. Your result will be solely a function named solve(), and do not call this function in your code.
Make sure the code is free of bug and can pass the test cases provided. You can use any library you want. The test cases are provided in the code. Do not call the solve() function in your code. 
Programming Problem: A. Line Trip
There is a road, which can be represented as a number line. You are located in the point $$$0$$$ of the number line, and you want to travel from the point $$$0$$$ to the point $$$x$$$, and back to the point $$$0$$$.
You travel by car, which spends $$$1$$$ liter of gasoline per $$$1$$$ unit of distance travelled. When you start at the point $$$0$$$, your car is fully fueled (its gas tank contains the maximum possible amount of fuel).
There are $$$n$$$ gas stations, located in points $$$a_1, a_2, \dots, a_n$$$. When you arrive at a gas station, you fully refuel your car.
Note that you can refuel only at gas stations, and there are no gas stations in points $$$0$$$ and $$$x$$$
.
You have to calculate the minimum possible volume of the gas tank in your car (in liters) that will allow you to travel from the point $$$0$$$ to the point $$$x$$$ and back to the point $$$0$$$.
Input
The first line contains one integer $$$t$$$ ($$$1 \le t \le 1000$$$) — the number of test cases.
Each test case consists of two lines:
the first line contains two integers $$$n$$$ and $$$x$$$ ($$$1 \le n \le 50$$$; $$$2 \le x \le 100$$$);
the second line contains $$$n$$$ integers $$$a_1, a_2, \dots, a_n$$$ ($$$0 < a_1 < a_2 < \dots < a_n < x$$$).
Output
For each test case, print one integer — the minimum possible volume of the gas tank in your car that will allow you to travel from the point $$$0$$$ to the point $$$x$$$ and back.
Example
Input
3
3 7
1 2 5
3 6
1 2 5
1 10
7
Output
4
3
7
Note
In the first test case of the example, if the car has a gas tank of $$$4$$$ liters, you can travel to $$$x$$$ and back as follows:
travel to the point $$$1$$$, then your car's gas tank contains $$$3$$$ liters of fuel;
refuel at the point $$$1$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$2$$$, then your car's gas tank contains $$$3$$$ liters of fuel;
refuel at the point $$$2$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$5$$$, then your car's gas tank contains $$$1$$$ liter of fuel;
refuel at the point $$$5$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$7$$$, then your car's gas tank contains $$$2$$$ liters of fuel;
travel to the point $$$5$$$, then your car's gas tank contains $$$0$$$ liters of fuel;
refuel at the point $$$5$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$2$$$, then your car's gas tank contains $$$1$$$ liter of fuel;
refuel at the point $$$2$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$1$$$, then your car's gas tank contains $$$3$$$ liters of fuel;
refuel at the point $$$1$$$, then your car's gas tank contains $$$4$$$ liters of fuel;
travel to the point $$$0$$$, then your car's gas tank contains $$$3$$$ liters of fuel.''',
]

for i, p in enumerate(prompts):
    prompt = "<|im_start|>system\nYou are DeepSeek. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
    stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS_THINKING,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    prompt += "<|im_start|>think"
    o = model.generate(
        prompt,
        sampling_params=sampling_params
    )
    ignore_str = "Wait"
    max_tokens_thinking_tmp = MAX_TOKENS_THINKING
    for i in range(NUM_IGNORE): # Num of times to skip stop token
        max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
        if max_tokens_thinking_tmp > 0:
            prompt += o[0].outputs[0].text + ignore_str
            sampling_params = SamplingParams(
                max_tokens=max_tokens_thinking_tmp,
                min_tokens=1,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=0.0,
            )
            o = model.generate(
                prompt,
                sampling_params=sampling_params
            )
    ### Final answer ###
    prompt += o[0].outputs[0].text # You can also append "Final Answer:" here like we do for some evaluations to prevent the model from just continuing to reason in its answer when early exiting
    stop_token_ids = tok("<|im_end|>")["input_ids"]
    sampling_params = SamplingParams(
        max_tokens=32768,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    o = model.generate(
        prompt,
        sampling_params=sampling_params,
    )
    print("With budget forcing:") # You will see that after the "Wait" in the reasoning trace it fixes its answer
    print(prompt + o[0].outputs[0].text)