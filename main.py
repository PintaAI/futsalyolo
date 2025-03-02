def calculate_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"

# Get test scores
math_score = 85
science_score = 92

# Calculate average
average = (math_score + science_score) / 2
print(f"Math score: {math_score}")
print(f"Science score: {science_score}")
print(f"Average score: {average}")

# Get letter grades
math_grade = calculate_grade(math_score)
science_grade = calculate_grade(science_score)

print(f"\nMath grade: {math_grade}")
print(f"Science grade: {science_grade}")
