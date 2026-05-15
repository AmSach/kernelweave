import subprocess
import time

print("Continuous Grind Mode Activated!")
print("This script will run test_agent.py in an infinite loop.")

while True:
    print("\n" + "="*50)
    print(f"Starting a new grind session at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    try:
        # Run test_agent.py and wait for it to complete
        subprocess.run(["python", "test_agent.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] test_agent.py exited with error: {e}")
    except Exception as e:
        print(f"\n[Error] Unexpected error: {e}")
        
    print("\nGrind session finished. Waiting 10 seconds before restarting...")
    time.sleep(10)
