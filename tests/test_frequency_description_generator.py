import unittest

from tools.generate_cub200_frequency_descriptions import (
    BANDS,
    build_user_prompt,
    canonicalize_and_validate_batch,
    parse_response_content,
    validate_class_payload,
)


class FrequencyDescriptionGeneratorTest(unittest.TestCase):
    def _valid_payload(self, class_name, candidates=2):
        cues = {
            "low": ("broad dark silhouette", "long balanced body"),
            "middle": ("compact head and narrow wings", "short neck and tapered tail"),
            "high": ("fine pale feather edges", "small sharply bounded markings"),
        }
        return {
            band: [
                "a photo of a {} with {}".format(class_name, cue)
                for cue in cues[band][:candidates]
            ]
            for band in BANDS
        }

    def test_valid_payload_is_canonicalized_to_dataset_name(self):
        class_name = "Black footed Albatross"
        response = {
            "Black-footed Albatross": self._valid_payload(class_name)
        }

        canonical, errors = canonicalize_and_validate_batch(
            [class_name], response, candidates=2, max_words=22
        )

        self.assertEqual(errors, [])
        self.assertEqual(list(canonical), [class_name])

    def test_non_visual_description_is_rejected(self):
        class_name = "Black footed Albatross"
        payload = self._valid_payload(class_name)
        payload["low"][0] = (
            "a photo of a Black footed Albatross found in an ocean habitat"
        )

        errors = validate_class_payload(
            class_name, payload, candidates=2, max_words=22
        )

        self.assertTrue(any("habitat" in error for error in errors))

    def test_policy_keyword_inside_class_name_is_not_rejected(self):
        class_name = "Song Sparrow"
        payload = self._valid_payload(class_name)

        errors = validate_class_payload(
            class_name, payload, candidates=2, max_words=22
        )

        self.assertEqual(errors, [])

    def test_sound_cue_outside_song_sparrow_name_is_rejected(self):
        class_name = "Song Sparrow"
        payload = self._valid_payload(class_name)
        payload["low"][0] = (
            "a photo of a Song Sparrow producing a distinctive melodic song"
        )

        errors = validate_class_payload(
            class_name, payload, candidates=2, max_words=22
        )

        self.assertTrue(any("sound information" in error for error in errors))

    def test_natural_an_article_is_accepted(self):
        class_name = "Indigo Bunting"
        payload = self._valid_payload(class_name)
        for band in BANDS:
            payload[band] = [
                description.replace(
                    "a photo of a Indigo Bunting",
                    "a photo of an Indigo Bunting",
                )
                for description in payload[band]
            ]

        errors = validate_class_payload(
            class_name, payload, candidates=2, max_words=22
        )

        self.assertEqual(errors, [])

    def test_trailing_json_comma_is_repaired(self):
        parsed = parse_response_content('{"bird": {"low": [],},}')

        self.assertEqual(parsed, {"bird": {"low": []}})

    def test_valid_classes_are_salvaged_from_partially_invalid_batch(self):
        first = "Black footed Albatross"
        second = "Indigo Bunting"
        invalid = self._valid_payload(second)
        invalid["high"] = []
        response = {
            first: self._valid_payload(first),
            second: invalid,
        }

        canonical, errors = canonicalize_and_validate_batch(
            [first, second], response, candidates=2, max_words=22
        )

        self.assertEqual(list(canonical), [first])
        self.assertTrue(any(second in error for error in errors))

    def test_generation_prompt_requests_exact_schema(self):
        prompt = build_user_prompt(
            ["Black footed Albatross"], candidates=5, max_words=22
        )

        self.assertIn('"Black footed Albatross"', prompt)
        self.assertIn('"middle"', prompt)
        self.assertIn("exactly 5 descriptions", prompt)


if __name__ == "__main__":
    unittest.main()
