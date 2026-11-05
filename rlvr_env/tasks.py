"""Task bank for the RLVR environment.

Each task is a dict with:
  id: unique string
  category: "code" or "math"
  difficulty: "easy" | "medium" | "hard"
  prompt: what the agent is shown
  function_name: (code tasks) the function the agent must define
  tests: (code tasks) list of (args_tuple, expected) pairs the verifier runs
  answer: (math tasks) the exact numeric answer
  reference_solution: a correct solution, used by the oracle baseline in
      evaluate.py and as documentation. The verifier never sees it.
"""

CODE_TASKS = [
    {
        "id": "code-01",
        "category": "code",
        "difficulty": "easy",
        "prompt": "Write a function `add(a, b)` that returns the sum of a and b.",
        "function_name": "add",
        "tests": [((1, 2), 3), ((-5, 5), 0), ((0, 0), 0), ((100, 250), 350)],
        "reference_solution": "def add(a, b):\n    return a + b\n",
    },
    {
        "id": "code-02",
        "category": "code",
        "difficulty": "easy",
        "prompt": "Write a function `is_prime(n)` that returns True if n is a prime number, else False. Treat numbers less than 2 as not prime.",
        "function_name": "is_prime",
        "tests": [((2,), True), ((3,), True), ((4,), False), ((17,), True), ((1,), False), ((0,), False), ((25,), False), ((97,), True)],
        "reference_solution": (
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
        ),
    },
    {
        "id": "code-03",
        "category": "code",
        "difficulty": "easy",
        "prompt": "Write a function `reverse_string(s)` that returns s reversed.",
        "function_name": "reverse_string",
        "tests": [(("hello",), "olleh"), (("",), ""), (("a",), "a"), (("racecar",), "racecar")],
        "reference_solution": "def reverse_string(s):\n    return s[::-1]\n",
    },
    {
        "id": "code-04",
        "category": "code",
        "difficulty": "easy",
        "prompt": (
            "Write a function `fizzbuzz_value(n)` that returns 'FizzBuzz' if n is divisible by both 3 and 5, "
            "'Fizz' if divisible by 3 only, 'Buzz' if divisible by 5 only, otherwise the string form of n."
        ),
        "function_name": "fizzbuzz_value",
        "tests": [((15,), "FizzBuzz"), ((3,), "Fizz"), ((5,), "Buzz"), ((7,), "7"), ((30,), "FizzBuzz"), ((1,), "1")],
        "reference_solution": (
            "def fizzbuzz_value(n):\n"
            "    if n % 15 == 0:\n"
            "        return 'FizzBuzz'\n"
            "    if n % 3 == 0:\n"
            "        return 'Fizz'\n"
            "    if n % 5 == 0:\n"
            "        return 'Buzz'\n"
            "    return str(n)\n"
        ),
    },
    {
        "id": "code-05",
        "category": "code",
        "difficulty": "easy",
        "prompt": "Write a function `factorial(n)` that returns n! for a non-negative integer n. factorial(0) is 1.",
        "function_name": "factorial",
        "tests": [((0,), 1), ((1,), 1), ((5,), 120), ((7,), 5040)],
        "reference_solution": (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        ),
    },
    {
        "id": "code-06",
        "category": "code",
        "difficulty": "medium",
        "prompt": "Write a function `fibonacci(n)` that returns the n-th Fibonacci number, with fibonacci(0) = 0 and fibonacci(1) = 1.",
        "function_name": "fibonacci",
        "tests": [((0,), 0), ((1,), 1), ((2,), 1), ((6,), 8), ((10,), 55), ((15,), 610)],
        "reference_solution": (
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
    },
    {
        "id": "code-07",
        "category": "code",
        "difficulty": "easy",
        "prompt": "Write a function `count_vowels(s)` that returns the number of vowels (a, e, i, o, u, case-insensitive) in s.",
        "function_name": "count_vowels",
        "tests": [(("hello",), 2), (("AEIOU",), 5), (("xyz",), 0), (("",), 0), (("Education",), 5)],
        "reference_solution": (
            "def count_vowels(s):\n"
            "    return sum(1 for ch in s.lower() if ch in 'aeiou')\n"
        ),
    },
    {
        "id": "code-08",
        "category": "code",
        "difficulty": "medium",
        "prompt": "Write a function `merge_sorted(a, b)` that takes two sorted lists of numbers and returns one sorted list containing all elements.",
        "function_name": "merge_sorted",
        "tests": [(([1, 3, 5], [2, 4, 6]), [1, 2, 3, 4, 5, 6]), (([], [1, 2]), [1, 2]), (([5], []), [5]), (([1, 1], [1]), [1, 1, 1])],
        "reference_solution": (
            "def merge_sorted(a, b):\n"
            "    out, i, j = [], 0, 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n"
            "            out.append(a[i]); i += 1\n"
            "        else:\n"
            "            out.append(b[j]); j += 1\n"
            "    return out + a[i:] + b[j:]\n"
        ),
    },
    {
        "id": "code-09",
        "category": "code",
        "difficulty": "medium",
        "prompt": (
            "Write a function `binary_search(arr, target)` that returns the index of target in the sorted list arr, "
            "or -1 if target is not present."
        ),
        "function_name": "binary_search",
        "tests": [(([1, 3, 5, 7, 9], 5), 2), (([1, 3, 5, 7, 9], 1), 0), (([1, 3, 5, 7, 9], 9), 4), (([1, 3, 5, 7, 9], 4), -1), (([], 1), -1)],
        "reference_solution": (
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        if arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
    },
    {
        "id": "code-10",
        "category": "code",
        "difficulty": "medium",
        "prompt": "Write a function `flatten(nested)` that takes a list that may contain nested lists (one level deep) and returns a flat list of all elements, in order.",
        "function_name": "flatten",
        "tests": [(([1, [2, 3], 4],), [1, 2, 3, 4]), (([],), []), (([[1, 2], [3]],), [1, 2, 3]), (([1, 2, 3],), [1, 2, 3])],
        "reference_solution": (
            "def flatten(nested):\n"
            "    out = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            out.extend(item)\n"
            "        else:\n"
            "            out.append(item)\n"
            "    return out\n"
        ),
    },
]

MATH_TASKS = [
    {
        "id": "math-01",
        "category": "math",
        "difficulty": "easy",
        "prompt": "What is 17 * 23 + 41? Give only the number.",
        "answer": 432.0,
        "reference_solution": "432",
    },
    {
        "id": "math-02",
        "category": "math",
        "difficulty": "easy",
        "prompt": "A shirt costs $40 and is discounted by 25%. What is the sale price in dollars? Give only the number.",
        "answer": 30.0,
        "reference_solution": "30",
    },
    {
        "id": "math-03",
        "category": "math",
        "difficulty": "easy",
        "prompt": "Solve for x: 3x + 7 = 22. Give only the number.",
        "answer": 5.0,
        "reference_solution": "5",
    },
    {
        "id": "math-04",
        "category": "math",
        "difficulty": "medium",
        "prompt": "What is the sum of the first 20 positive integers (1 + 2 + ... + 20)? Give only the number.",
        "answer": 210.0,
        "reference_solution": "210",
    },
    {
        "id": "math-05",
        "category": "math",
        "difficulty": "medium",
        "prompt": "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Give only the number.",
        "answer": 72.0,
        "reference_solution": "72",
    },
    {
        "id": "math-06",
        "category": "math",
        "difficulty": "medium",
        "prompt": "What is 15% of 240? Give only the number.",
        "answer": 36.0,
        "reference_solution": "36",
    },
    {
        "id": "math-07",
        "category": "math",
        "difficulty": "medium",
        "prompt": "Solve for x: 2(x - 3) = 4x - 10. Give only the number.",
        "answer": 2.0,
        "reference_solution": "2",
    },
    {
        "id": "math-08",
        "category": "math",
        "difficulty": "hard",
        "prompt": "What is the greatest common divisor (GCD) of 84 and 126? Give only the number.",
        "answer": 42.0,
        "reference_solution": "42",
    },
    {
        "id": "math-09",
        "category": "math",
        "difficulty": "hard",
        "prompt": "A rectangle has length 12 and width 5. What is the length of its diagonal? Give only the number.",
        "answer": 13.0,
        "reference_solution": "13",
    },
    {
        "id": "math-10",
        "category": "math",
        "difficulty": "hard",
        "prompt": "What is 2^10 - 2^8? Give only the number.",
        "answer": 768.0,
        "reference_solution": "768",
    },
]

TASKS = CODE_TASKS + MATH_TASKS


def get_task(task_id):
    for task in TASKS:
        if task["id"] == task_id:
            return task
    raise KeyError(f"unknown task id: {task_id}")


def list_tasks(category=None):
    if category is None:
        return list(TASKS)
    return [t for t in TASKS if t["category"] == category]
