"""Bare whole numbers in a table row that also holds an amount with a decimal comma.

RapidOCR loses the comma of an amount in a form's cells ("100,00" read as "10000") at a
score of 1.0, so neither mean nor p10 shows it. All values here are made up.
"""

from docflow.domain.models import OcrLine
from docflow.pipeline.amounts import bare_numbers_beside_amounts


def cell(text, left, top, height=40):
    return OcrLine(text=text, score=1.0, box=(left, top, left + 120, top + height))


def test_bare_numbers_in_a_row_with_an_amount_are_counted():
    lines = [
        cell("Summe", 100, 500),
        cell("10000", 300, 502),
        cell("25000", 500, 498),
        cell("12,50", 900, 500),
    ]

    assert bare_numbers_beside_amounts(lines) == 2


def test_a_bare_number_in_another_row_is_not_counted():
    lines = [cell("10115", 300, 200), cell("Berlin", 500, 200), cell("12,50", 900, 500)]

    assert bare_numbers_beside_amounts(lines) == 0


def test_a_row_overlapping_less_than_half_a_line_height_is_another_row():
    lines = [cell("10000", 300, 500), cell("12,50", 900, 521)]

    assert bare_numbers_beside_amounts(lines) == 0


def test_amounts_with_thousands_dots_currency_or_sign_count_as_amounts():
    for amount in ("1.234,56", "12,50 €", "EUR 12,50", "-3,00"):
        lines = [cell("10000", 300, 500), cell(amount, 900, 500)]

        assert bare_numbers_beside_amounts(lines) == 1, amount


def test_years_ids_and_short_numbers_are_not_amounts_that_lost_a_comma():
    lines = [
        cell("2024", 100, 500),
        cell("12345678", 250, 500),
        cell("100", 400, 500),
        cell("10000 Stk", 550, 500),
        cell("12,50", 900, 500),
    ]

    assert bare_numbers_beside_amounts(lines) == 0


def test_lines_without_a_box_are_left_out():
    lines = [OcrLine(text="10000", score=1.0), cell("12,50", 900, 500)]

    assert bare_numbers_beside_amounts(lines) == 0
