
def calculate_ratio(a, b):
    if b == 0:
        return 'Error: Division by zero is not allowed'
    return a / b

if __name__ == "__main__":
    print("Starting calculation...")
    # Updated denominator to avoid division by zero
    result = calculate_ratio(10, 2)
    print(f"Result: {result}")