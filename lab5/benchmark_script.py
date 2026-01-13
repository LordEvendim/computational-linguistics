import ollama
import json
import time
from datetime import datetime

MODELS = {
    "small": "gemma2:2b",
    "large_reasoning": "qwen3:14b"
}

TASKS = {
    "1_instruction_following": {
        "name": "Instruction Following (Complex Constraints)",
        "prompt": """Write a list of 3 tips for staying healthy.
Strict Rules:
1. Use exactly 3 bullet points.
2. The second bullet point must contain the word 'submarine'.
3. Do not use any words longer than 7 letters. Keep it simple.""",
        "category": "Instruction Following"
    },
    "2_logical_reasoning": {
        "name": "Logical Reasoning (Deductive Logic)",
        "prompt": """Four colleagues—Alice, Bob, Charlie, and Diana—are sitting in a row of four chairs numbered 1 to 4 from left to right.
1. Alice is not in chair 1 or 4.
2. Bob is sitting immediately to the left of Charlie.
3. Diana is sitting somewhere to the right of Alice.
Who is sitting in Chair 2? Explain your reasoning step-by-step.""",
        "category": "Logical Reasoning"
    },
    "3_creative_writing": {
        "name": "Creative Writing (Style Transfer & Tone)",
        "prompt": """Describe the mundane act of washing dirty dishes after a dinner party, but write it in the style of a gritty, 1940s Film Noir detective monologue. Focus on the grease, the water, and the regret.""",
        "category": "Creative Writing"
    },
    "4_code_generation": {
        "name": "Code Generation (Algorithmic Logic)",
        "prompt": """Write a Python function called `find_palindromic_primes(n)`. The function should return a list of all numbers up to `n` that are **both** prime numbers and palindromes (read the same forwards and backwards). Example: 131 is a palindromic prime.""",
        "category": "Code Generation"
    },
    "5_reading_comprehension": {
        "name": "Reading Comprehension (Inference)",
        "prompt": """**Read this passage:** 'I was absolutely thrilled when Mark got the promotion. Really. It’s great that he’ll be my boss now, considering I trained him three years ago. I’m just so happy I can finally relax and let him take all the credit while I stay in this cubicle. It’s what I’ve always wanted, honestly.'
**Question:** Based on the text, how does the narrator *actually* feel about Mark's promotion? Explain which specific phrases betray their true feelings.""",
        "category": "Reading Comprehension"
    },
    "6_common_sense": {
        "name": "Common Sense Reasoning (Physical World)",
        "prompt": """I put a wet t-shirt inside a freezer and leave it there for 24 hours. Then, I take it out and immediately put it into a microwave on high for 30 seconds. Describe exactly what happens to the t-shirt and its texture at that moment. Is it dry? Is it hot?""",
        "category": "Common Sense Reasoning"
    },
    "7_language_understanding": {
        "name": "Language Understanding (Ambiguity)",
        "prompt": """Explain the meaning of this sentence: 'The old man the boat.'
Identify the verb in this sentence and explain what the sentence is actually saying about the people involved.""",
        "category": "Language Understanding"
    },
    "8_factual_knowledge": {
        "name": "Factual Knowledge (Hallucination Check)",
        "prompt": """Compare and contrast the contributions of Nikola Tesla and Thomas Edison regarding electrical current. Did they ever work together directly? If so, describe the nature of their professional breakup.""",
        "category": "Factual Knowledge"
    },
    "9_math_problem_solving": {
        "name": "Mathematical Problem Solving (The 'Trick' Question)",
        "prompt": """A snail is at the bottom of a 20-foot well. Each day, it climbs up 5 feet, but at night, while sleeping, it slides back down 4 feet. How many days will it take for the snail to reach the top of the well?""",
        "category": "Mathematical Problem Solving"
    },
    "10_ethical_reasoning": {
        "name": "Ethical Reasoning & Nuance",
        "prompt": """You are a senior mentor. A junior employee has just pitched a 'revolutionary' AI idea that is actually technically impossible and legally dangerous. Write a response to them that shuts down the project immediately but maintains their enthusiasm for innovation. Do not be mean, but do not be vague.""",
        "category": "Ethical Reasoning"
    }
}

FEW_SHOT_EXAMPLES = {
    "Instruction Following": """Example 1:
Task: Write a list of 2 cleaning chores.
Rule 1: Use exactly 2 bullet points.
Rule 2: The first bullet must contain the word 'soap'.
Rule 3: No words longer than 5 letters.
Output:
- Buy soap
- Wash rug

Example 2:
Task: Write a movie list with 2 items.
Rule 1: Exact 2 bullets.
Rule 2: Second item must be 'Jaws'.
Rule 3: No words longer than 4 letters.
Output:
- Dune
- Jaws
""",
    "Logical Reasoning": """Example 1:
Puzzle: Red is faster than Blue. Blue is faster than Green. Is Red faster than Green?
Answer: Yes. If Red > Blue and Blue > Green, then Red > Green.

Example 2:
Puzzle: Cat is left of Dog. Dog is left of Mouse. Who is in the middle?
Answer: The Dog is in the middle. (Order: Cat, Dog, Mouse).
""",
    "Creative Writing": """Example 1:
Task: Describe making toast in the style of a medieval knight.
Output: I summon the fire! The bread enters the iron gate. It burns! Victory is mine, a crunch worthy of a king.

Example 2:
Task: Describe brushing teeth in the style of a frantic sports commentator.
Output: He loads the brush! He's going in for the molars! The foaming action is incredible! He spits! A flawless finish!
""",
    "Code Generation": """Example 1:
Task: Write a python function `is_even_and_positive(n)` that returns True if n is even and > 0.
Output:
def is_even_and_positive(n):
    return n > 0 and n % 2 == 0

Example 2:
Task: Write a python function `sum_of_squares(n)` that returns sum of squares from 1 to n.
Output:
def sum_of_squares(n):
    return sum(i * i for i in range(1, n + 1))
""",
    "Reading Comprehension": """Example 1:
Passage: "Oh great, another flat tire," he laughed, kicking the wheel.
Question: Is he happy?
Answer: No. The 'laughed' combined with 'kicking the wheel' and 'Oh great' indicates sarcastic frustration.

Example 2:
Passage: The cake was dry, burnt, and hard as a rock. "Delicious," she whispered, not swallowing.
Question: Did she like the cake?
Answer: No. 'Not swallowing' and the description of the cake contradict her labeled praise, implying she is lying to be polite or afraid.
""",
    "Common Sense Reasoning": """Example 1:
Question: If I drop a raw egg on a concrete floor, what happens?
Answer: It smashes and creates a mess because raw eggs are fragile and concrete is hard.

Example 2:
Question: I put a snowball in a frying pan on high heat. What happens after 10 minutes?
Answer: It melts into water and then evaporates into steam. It will be gone or just a dry hot pan.
""",
    "Language Understanding": """Example 1:
Sentence: "The bandage was wound around the wound."
Explanation: "Wound" (1st) is past tense of wind (to wrap). "Wound" (2nd) is an injury.

Example 2:
Sentence: "The complex houses married and single soldiers."
Explanation: "Complex" is the subject (housing complex). "Houses" is the verb (to provide housing).
""",
    "Factual Knowledge": """Example 1:
Question: Compare Isaac Newton and Leibniz regarding calculus.
Answer: Both independently discovered calculus. There was a bitter dispute over priority, but we use Leibniz's notation today.

Example 2:
Question: Did John Lennon and Paul McCartney ever collaborate?
Answer: Yes, they were the primary songwriting partnership of The Beatles and are one of the most successful duos in history.
""",
    "Mathematical Problem Solving": """Example 1:
Question: I have 10 apples. I eat 2, drop 1, and buy 5 more. How many do I have?
Answer: 10 - 2 - 1 + 5 = 12 apples.

Example 2:
Question: A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball?
Answer: 5 cents. (Ball = x. Bat = x+1. x + x+1 = 1.10 => 2x = 0.10 => x = 0.05).
""",
    "Ethical Reasoning": """Example 1:
Task: A friend asks to copy your homework. Refuse but be nice.
Response: "I can't let you copy the answers because we'll both get in trouble, but I can help you understand the questions if you want."

Example 2:
Task: An employee asks for a raise but company is firing people. Deny but keep morale.
Response: "We cannot offer raises right now due to the budget freezing, but I value your work and we can revisit this when the financial situation stabilizes."
"""
}

def construct_prompt(strategy, task_key, task_data):
    base_prompt = task_data["prompt"]
    category = task_data["category"]
    
    if strategy == "zero-shot":
        return base_prompt
        
    elif strategy == "few-shot":
        examples = FEW_SHOT_EXAMPLES.get(category, "")
        if not examples:
            return f"{base_prompt}"
        return f"Here are some examples:\n\n{examples}\nNow solve this:\n{base_prompt}"
        
    elif strategy == "chain-of-thought":
        return f"{base_prompt}\n\nLet's solve this step by step:\n1. First,"
    
    return base_prompt

def run_benchmark():
    results = []
    
    try:
        ollama.list()
    except Exception as e:
        print("Error: Could not connect to Ollama")
        print(f"Details: {e}")
        return

    model_small = MODELS["small"]
    print(f"   Benchmarking Small Model: {model_small}")
    
    strategies_small = ["zero-shot", "few-shot", "chain-of-thought"]
    
    for task_key, task_data in TASKS.items():
        for strategy in strategies_small:
            print(f"Running Task: {task_data['name']} [{strategy}]")
            prompt = construct_prompt(strategy, task_key, task_data)
            
            start_time = time.time()
            try:
                response = ollama.generate(model=model_small, prompt=prompt)
                output = response['response']
                duration = time.time() - start_time
                
                results.append({
                    "model": model_small,
                    "model_type": "small_standard",
                    "task": task_data["name"],
                    "strategy": strategy,
                    "prompt_used": prompt,
                    "output": output,
                    "duration_seconds": round(duration, 2),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({
                    "model": model_small,
                    "error": str(e),
                    "task": task_data["name"],
                    "strategy": strategy
                })

    model_large = MODELS["large_reasoning"]
    print(f"   Benchmarking Large Model: {model_large}")
    
    strategies_large = ["zero-shot", "few-shot"]
    
    for task_key, task_data in TASKS.items():
        for strategy in strategies_large:
            print(f"Running Task: {task_data['name']} [{strategy}]")
            prompt = construct_prompt(strategy, task_key, task_data)
            
            start_time = time.time()
            try:
                response = ollama.generate(model=model_large, prompt=prompt)
                output = response['response']
                duration = time.time() - start_time
                
                results.append({
                    "model": model_large,
                    "model_type": "large_reasoning",
                    "task": task_data["name"],
                    "strategy": strategy,
                    "prompt_used": prompt,
                    "output": output,
                    "duration_seconds": round(duration, 2),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({
                    "model": model_large,
                    "error": str(e),
                    "task": task_data["name"],
                    "strategy": strategy
                })

    filename = f"benchmark_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_benchmark()
