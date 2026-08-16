import pytest
from src.prompts import (NUMBER_WORDS, DEFAULT_OBJECTS, pluralize,
                         build_prompt, PromptSpec, generate_grid)

def test_singular_no_pluralize():
    assert build_prompt(1, "apple") == "one apple"

def test_plural_regular():
    assert build_prompt(3, "cat") == "three cats"

def test_pluralize_edge_cases():
    assert pluralize("bus") == "buses"
    assert pluralize("berry") == "berries"
    assert pluralize("car") == "cars"

def test_count_out_of_range_raises():
    with pytest.raises(ValueError):
        build_prompt(11, "cat")
    with pytest.raises(ValueError):
        build_prompt(0, "cat")

def test_high_counts_supported():
    assert build_prompt(8, "cat") == "eight cats"
    assert build_prompt(10, "dog") == "ten dogs"

def test_generate_grid_shape_and_content():
    counts, objects, seeds = [1, 2], ["cat", "bus"], [0, 1, 2]
    grid = generate_grid(counts, objects, seeds)
    assert len(grid) == 2 * 2 * 3
    assert all(isinstance(p, PromptSpec) for p in grid)
    one_cat = [p for p in grid if p.count == 1 and p.obj == "cat"]
    assert len(one_cat) == 3
    assert one_cat[0].text == "one cat"
    assert {p.seed for p in one_cat} == {0, 1, 2}

def test_default_objects_are_seven_plus_and_countable():
    assert len(DEFAULT_OBJECTS) >= 7
    assert len(NUMBER_WORDS) == 10
