import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.vetting_agent import VettingAgent

def main():
    print("--- Manufacturer Compliance Intelligence & Vetting Agent ---")
    agent = VettingAgent()
    
    while True:
        print("\nOptions:")
        print("1. Vet a Manufacturer")
        print("2. Exit")
        choice = input("Select an option: ")
        
        if choice == '1':
            name = input("Enter manufacturer name: ")
            location = input("Enter location (optional, press Enter to skip): ")
            
            print("\n--- Vetting Process Started ---")
            assessment = agent.vet_manufacturer(name, location if location else None)
            
            print("\n--- Assessment Results ---")
            print(f"Manufacturer: {assessment.manufacturer.name}")
            print(f"Risk Score: {assessment.risk_score}/100")
            print(f"Recommendation: {assessment.recommendation}")
            print(f"Explanation: {assessment.explanation}")
            print(f"Confidence: {assessment.confidence_score}")
            
            while True:
                print("\nOptions:")
                print("1. Ask a follow-up question")
                print("2. Return to main menu")
                sub_choice = input("Select an option: ")
                
                if sub_choice == '1':
                    question = input("Enter your question: ")
                    answer = agent.discuss_assessment(assessment, question)
                    print(f"\nAgent: {answer}")
                elif sub_choice == '2':
                    break
                else:
                    print("Invalid option.")

        elif choice == '2':
            print("Exiting...")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
