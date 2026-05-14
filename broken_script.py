def calculate_ratio(a, b):
    return a / b

if __name__ == "__main__":
    print("Starting calculation...")
    result = calculate_ratio(10, 0) # Bug: Division by zero!
    print(f"Result: {result}")
