import unittest
from streamlit_meal_planner import calculate_bmr

class TestMealPlanner(unittest.TestCase):

    def test_calculate_bmr_male(self):
        # Test calculate_bmr for a male
        weight = 70
        height = 175
        age = 30
        gender = "Male"
        expected_bmr = 1671.25  # You should replace this with the actual expected result
        result = calculate_bmr(weight, height, age, gender)
        self.assertAlmostEqual(result, expected_bmr, delta=0.01)  # Adjust delta as needed

    def test_calculate_bmr_female(self):
        # Test calculate_bmr for a female
        weight = 60
        height = 160
        age = 25
        gender = "Female"
        expected_bmr = 1384.6  # You should replace this with the actual expected result
        result = calculate_bmr(weight, height, age, gender)
        self.assertAlmostEqual(result, expected_bmr, delta=0.01)  # Adjust delta as needed

if __name__ == '__main__':
    unittest.main()
