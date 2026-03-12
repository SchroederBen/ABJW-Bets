from PyScripts.getTeams import get_all_teams
from AI_APIs import summarize_teams_with_ai

def main():
    teams = get_all_teams()

    print("=== RAW DB TEAM COUNT ===")
    print(len(teams))

    ai_response = summarize_teams_with_ai(teams)

    print("\n=== OPENAI RESPONSE ===")
    print(ai_response)

if __name__ == "__main__":
    main()