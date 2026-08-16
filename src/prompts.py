from dataclasses import dataclass

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

DEFAULT_OBJECTS = ["apple", "cat", "car", "bird", "bottle",
                   "chair", "cup", "dog", "banana", "clock"]


def pluralize(noun: str) -> str:
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def build_prompt(count: int, obj: str) -> str:
    if count not in NUMBER_WORDS:
        raise ValueError(f"count {count} out of supported range 1-10")
    word = NUMBER_WORDS[count]
    noun = obj if count == 1 else pluralize(obj)
    return f"{word} {noun}"


@dataclass(frozen=True)
class PromptSpec:
    count: int
    obj: str
    seed: int
    text: str


def generate_grid(counts, objects, seeds):
    grid = []
    for obj in objects:
        for count in counts:
            text = build_prompt(count, obj)
            for seed in seeds:
                grid.append(PromptSpec(count=count, obj=obj, seed=seed, text=text))
    return grid
