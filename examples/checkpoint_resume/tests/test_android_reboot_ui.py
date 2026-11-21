import unittest
import xml.etree.ElementTree as ET
from android_reboot import checkbox_for_label, visible_node


def node(text='', kind='android.widget.TextView', bounds='[180,600][700,660]'):
    return ET.Element('node', {'text': text, 'class': kind, 'bounds': bounds, 'checked': 'true'})


class CheckboxLookupTest(unittest.TestCase):
    def test_hidden_menu_text_is_not_tappable(self):
        self.assertFalse(visible_node(node('task', bounds='[0,0][0,0]')))
        self.assertTrue(visible_node(node('task')))

    def test_compose_rounding_at_touching_edges(self):
        label = node('charging', bounds='[189,611][681,674]')
        box = node(kind='android.widget.CheckBox', bounds='[64,612][190,738]')
        self.assertIs(checkbox_for_label([label, box], 'charging'), box)

    def test_one_visible_checkbox_is_sufficient(self):
        label = node('charging')
        box = node(kind='android.widget.CheckBox', bounds='[60,580][180,700]')
        self.assertIs(checkbox_for_label([label, box], 'charging'), box)

    def test_other_row_and_hidden_nodes_do_not_match(self):
        label = node('charging')
        other = node(kind='android.widget.CheckBox', bounds='[60,300][180,420]')
        hidden = node(kind='android.widget.CheckBox', bounds='[0,0][0,0]')
        self.assertIsNone(checkbox_for_label([label, other, hidden], 'charging'))

    def test_ambiguous_or_hidden_label_fails_closed(self):
        box = node(kind='android.widget.CheckBox', bounds='[60,580][180,700]')
        self.assertIsNone(checkbox_for_label([node('charging'), node('charging'), box], 'charging'))
        self.assertIsNone(checkbox_for_label([node('charging', bounds='[0,0][0,0]'), box], 'charging'))


if __name__ == '__main__':
    unittest.main()
