import sys
import os
from src.agent.vetting_agent import VettingAgent

def clean_exit():
    print("\nExiting Vetting Agent. Goodbye!")
    sys.exit(0)

def main():
    print("==================================================")
    print("   MANUFACTURER COMPLIANCE INTELLIGENCE AGENT")
    print("==================================================")
    
    agent = VettingAgent()
    if not agent.client:
        print("Note: AI features disabled (GOOGLE_API_KEY missing).")

    while True:
        print("\nEnter manufacturer name to vet (or 'q' to quit):")
        name = input("> ").strip()
        
        if name.lower() in ['q', 'quit', 'exit']:
            clean_exit()
            
        if not name:
            continue
            
        print(f"\n🔍 Investigating '{name}'... This may take a moment.")
        
        try:
            assessment = agent.vet_manufacturer(name)
            
            print("\n" + "="*50)
            print(f"ASSESSMENT REPORT: {assessment.manufacturer.name}")
            print("="*50)
            print(f"Risk Score:    {assessment.risk_score} / 100")
            print(f"Status:        {assessment.recommendation.upper()}")
            print(f"Confidence:    {assessment.confidence_score * 100:.0f}%")
            print("-" * 50)
            print("DETAILS:")
            print(assessment.explanation)
            print("="*50)
            
            if agent.client:
                print("\n💬 Interactive Session (Type 'exit' to vet another company)")
                print(f"Ask questions about {name}'s compliance history.")
                
                while True:
                    q = input(f"\nYour question about {name} > ").strip()
                    if q.lower() in ['exit', 'back', 'new']:
                        break
                    if q.lower() in ['q', 'quit']:
                        clean_exit()
                        
                    if not q:
                        continue
                        
                    print("Thinking...")
                    answer = agent.discuss_assessment(assessment, q)
                    print(f"\n🤖 Agent: {answer}")
            else:
                input("\nPress Enter to continue...")

        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
