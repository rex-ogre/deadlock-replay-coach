from __future__ import annotations

from deadlock_coach.names import LANE_COLORS, Names


class TestFallbacks:
    def test_unknown_hero_renders_rather_than_raising(self):
        """A demo from a newer patch will contain heroes this table has never
        seen; the report must still render."""
        assert Names().hero(9999) == "Hero#9999"

    def test_unknown_team_and_ability(self):
        assert Names().team(7) == "Team#7"
        assert Names().ability(123) == "Item#123"

    def test_none_ids_are_not_errors(self):
        names = Names()
        assert names.hero(None) == "unknown"
        assert names.team(None) == "unknown"
        assert names.lane(None) == "unknown"


class TestLookups:
    def test_known_names_win(self, names):
        assert names.hero(1) == "Infernus"
        assert names.team(2) == "Hidden King"

    def test_lane_colors(self):
        names = Names()
        assert names.lane(1) == "yellow"
        assert names.lane(6) == "purple"
        assert set(LANE_COLORS) == {0, 1, 3, 4, 6}

    def test_objective_labels_are_title_cased(self):
        assert Names().objective("guardian") == "Guardian"
        assert Names().objective("BASE_GUARDIAN") == "Base Guardian"

    def test_unknown_objective_type_passes_through_unchanged(self):
        assert Names().objective("some_new_thing") == "some_new_thing"

    def test_missing_objective_type_has_a_default(self):
        assert Names().objective(None) == "Objective"


def test_from_boon_populates_tables_when_available():
    """Skipped implicitly if boon is absent — `from_boon` must not raise either way."""
    names = Names.from_boon()
    assert isinstance(names.heroes, dict)
    if names.heroes:
        assert names.hero(1) != "Hero#1"
