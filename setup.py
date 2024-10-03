import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("Setting up the Sentiment Analysis project...")

    # List of required packages
    packages = ['openai', 'python-dotenv']

    # Install required packages
    for package in packages:
        print(f"Installing {package}...")
        install(package)

    # Create the results directory if it doesn't exist
    if not os.path.exists('results'):
        print("Creating 'results' directory...")
        os.makedirs('results')

    # Create a sample .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("Creating sample .env file...")
        with open('.env', 'w') as f:
            f.write("OPENAI_API_KEY=your_api_key_here")
        print("Please update the .env file with your actual OpenAI API key.")

    print("\nSetup complete! You can now run the sentiment analysis script.")
    print("Make sure to update the .env file with your actual OpenAI API key before running the main script.")

if __name__ == "__main__":
    main()