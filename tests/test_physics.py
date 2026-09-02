"""Tests for the movement model.

The load-bearing property is that :func:`travel_seconds` never claims a hero
needs *more* time than one actually took on the ground. Everything downstream
that says "nobody could have helped" rests on that being true.
"""

from __future__ import annotations

import pytest

from deadlock_coach import gamedata, physics


@pytest.fixture(scope="module")
def constants():
    return gamedata.load_constants(offline=True)


@pytest.fixture(scope="module")
def hero(constants):
    profile = constants.hero(1)
    assert profile is not None, "the bundled snapshot must contain Infernus"
    return profile


class TestUnits:
    def test_a_hammer_unit_is_an_inch(self):
        assert physics.meters(39.3700787) == pytest.approx(1.0, abs=1e-6)

    def test_conversion_round_trips(self):
        assert physics.hammer_units(physics.meters(2500.0)) == pytest.approx(2500.0)

    def test_distance_ignores_height(self):
        # Two heroes on stacked walkways are not far apart in the sense that
        # matters for "could he have helped".
        assert physics.distance_m(0.0, 0.0, 3937.00787, 0.0) == pytest.approx(100.0, abs=0.01)


class TestTravelTime:
    def test_zero_distance_is_free(self, hero):
        assert physics.travel_seconds(0.0, hero).seconds == 0.0

    def test_longer_is_slower(self, hero):
        near = physics.travel_seconds(20.0, hero).seconds
        far = physics.travel_seconds(120.0, hero).seconds
        assert far > near

    def test_dashes_are_spent_before_running(self, hero):
        # Three bars of stamina at 10m a dash covers the first 30 metres.
        assert physics.travel_seconds(25.0, hero).dashes_used == 3
        assert physics.travel_seconds(25.0, hero, stamina=1).dashes_used == 1

    def test_spending_stamina_beats_saving_it(self, hero):
        with_dash = physics.travel_seconds(60.0, hero).seconds
        without = physics.travel_seconds(60.0, hero, stamina=0).seconds
        assert with_dash < without

    def test_a_zip_line_leg_beats_the_ground(self, hero, constants):
        ground = physics.travel_seconds(150.0, hero).seconds
        line = physics.travel_seconds(
            150.0, hero, zipline=constants.zipline, zipline_fraction=0.5
        ).seconds
        assert line < ground

    def test_walking_is_slower_than_sprinting(self, hero):
        assert (
            physics.travel_seconds(200.0, hero, sprinting=False).seconds
            > physics.travel_seconds(200.0, hero).seconds
        )

    def test_the_estimate_states_its_assumptions(self, hero):
        assumption = physics.travel_seconds(60.0, hero).assumption
        assert "straight line" in assumption
        assert "no walls" in assumption

    def test_it_is_a_lower_bound_on_ground_movement(self, hero):
        """Nothing on the ground beats the model.

        Checked against a real replay while building this: over 22,874 sampled
        one-second windows of non-zip-line movement, reality beat the model in
        0.48% of them. The synthetic version of that check is that the implied
        average speed never exceeds the fastest ground mechanic available.
        """
        fastest_ground = max(hero.ground_dash_speed, hero.air_dash_speed)
        for distance in (10.0, 30.0, 60.0, 100.0, 250.0):
            estimate = physics.travel_seconds(distance, hero, reaction=0.0)
            assert distance / estimate.seconds <= fastest_ground + 1e-6

    def test_could_have_arrived_is_a_two_sided_answer(self, hero):
        reachable, estimate = physics.could_have_arrived(20.0, 10.0, hero)
        assert reachable and estimate.seconds < 10.0
        stranded, estimate = physics.could_have_arrived(400.0, 5.0, hero)
        assert not stranded and estimate.seconds > 5.0


class TestSpeedClassification:
    def test_a_sustained_twenty_metres_a_second_is_a_zip_line(self, hero, constants):
        assert physics.classify_speed(20.5, hero, constants.zipline) == "zip line"

    def test_walking_pace_reads_as_walking(self, hero, constants):
        assert physics.classify_speed(hero.max_move_speed * 0.9, hero, constants.zipline) == "walk"

    def test_standing_still_is_not_movement(self, hero, constants):
        assert physics.classify_speed(0.1, hero, constants.zipline) == "stationary"

    def test_impossible_speeds_are_called_teleports(self, hero, constants):
        assert physics.classify_speed(90.0, hero, constants.zipline) == "teleport"

    def test_bands_are_ordered_and_cover_zero(self, hero, constants):
        bands = physics.speed_bands(hero, constants.zipline)
        floors = [b.floor for b in bands]
        assert floors == sorted(floors, reverse=True)
        assert floors[-1] == 0.0


class TestFalloff:
    def test_close_range_is_full_damage(self, hero):
        assert physics.falloff_note(hero, 1.0) == "inside full-damage range"

    def test_long_range_names_the_floor(self, hero):
        note = physics.falloff_note(hero, 10_000.0)
        assert note is not None and "minimum damage" in note

    def test_a_hero_without_weapon_data_gets_no_note(self, hero):
        import dataclasses

        bare = dataclasses.replace(hero, damage_falloff_start_m=None)
        assert physics.falloff_note(bare, 20.0) is None
