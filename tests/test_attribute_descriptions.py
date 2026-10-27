"""Attribute selection is fixed, conservative, unique, and honest about unknowns."""

import unittest

from utils.attribute_descriptions import GROUPS, build_attribute_manifest, inspect_sentence


def sentences(name):
    return [f'A {name} has a slender body.', f'A {name} has a compact body.',
            f'A {name} has a hooked bill.', f'A {name} has a forked tail.',
            f'A {name} is predominantly blue.', f'A {name} is predominantly red.',
            f'A {name} has broad stripes.', f'A {name} has fine speckles.']


class AttributeDescriptionsTest(unittest.TestCase):
    def test_actual_content_grouping_is_deterministic_and_has_no_duplicate_fillers(self):
        bank = {'base': sentences('base') + ['A base nests in the Pacific.', 'A base is 20 inches long.']}
        a = build_attribute_manifest(bank, ['base'])
        self.assertEqual(a, build_attribute_manifest(bank, ['base']))
        cls = a['classes'][0]
        self.assertEqual(cls['group_counts'], dict.fromkeys(GROUPS, 2))
        self.assertEqual(len({x['text'] for g in GROUPS for x in cls['groups'][g]}), 8)
        self.assertEqual(cls['ordinary_count'], 8)
        self.assertEqual(len(cls['rejected']), 2)
        self.assertEqual(a['coverage']['complete_attribute_classes'], 1)

    def test_species_name_does_not_create_color_or_pattern_evidence(self):
        self.assertEqual(inspect_sentence('A Red spotted Bird.', 'Red spotted Bird')[0], [])
        groups, reason = inspect_sentence('A Red-spotted Bird has a hooked bill.', 'Red spotted Bird')
        self.assertEqual(groups, ['part'])
        self.assertIsNone(reason)

    def test_missing_groups_remain_missing_and_future_names_are_not_required(self):
        manifest = build_attribute_manifest({'base': ['A base has a hooked bill.'],
                                             'future': {'invalid_future_payload': True}}, ['base'])
        cls = manifest['classes'][0]
        self.assertEqual(cls['group_counts'], {'shape': 0, 'part': 1, 'color': 0, 'pattern': 0})
        self.assertEqual(manifest['coverage']['complete_attribute_classes'], 0)
        self.assertNotIn('future', str(manifest))

    def test_explicit_lists_can_mark_unknown_and_do_not_fill_from_images_or_ordinary(self):
        supplied = {'base': {'shape': [], 'part': ['A base has a forked tail.'], 'color': [], 'pattern': []}}
        manifest = build_attribute_manifest({'base': sentences('base')}, ['base'], supplied)
        self.assertEqual(manifest['selection'], 'explicit_fixed_group_lists')
        self.assertEqual(manifest['classes'][0]['group_counts'], {'shape': 0, 'part': 1, 'color': 0, 'pattern': 0})

    def test_nonvisual_explicit_content_and_duplicated_sentences_are_rejected(self):
        supplied = {'base': {'shape': ['A base weighs 20 pounds.'], 'part': [], 'color': [], 'pattern': []}}
        with self.assertRaisesRegex(ValueError, 'rejected'):
            build_attribute_manifest({'base': sentences('base')}, ['base'], supplied)
        supplied['base']['shape'] = ['A base has a slender body.']
        supplied['base']['part'] = list(supplied['base']['shape'])
        with self.assertRaisesRegex(ValueError, 'unique'):
            build_attribute_manifest({'base': sentences('base')}, ['base'], supplied)

    def test_ambiguous_missing_or_invalid_inputs_fail_early(self):
        for bank, names in [({'base': []}, ['missing']),
                            ({'base-a': [], 'base a': []}, ['base a']),
                            ({'base': [True]}, ['base'])]:
            with self.subTest(bank=bank), self.assertRaises(ValueError):
                build_attribute_manifest(bank, names)


if __name__ == '__main__':
    unittest.main()
