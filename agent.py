import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# 1. Setup API Keys
# 2. Get keys safely (If they aren't in .env, this will be None)
GROQ_KEY = os.getenv("GROQ_API_KEY")
SERPER_KEY = os.getenv("SERPER_API_KEY")

# 2. Define the "Brain" (LLM) 
# We use the 'groq/' prefix so CrewAI knows exactly which provider to call
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")
)

# 3. Initialize Search Tools
search_tool = SerperDevTool()

# AGENT 1: The "Red" Mitigator
red_agent = Agent(
    role='Red Team Security Researcher',
    goal='Identify how an IAM role with "iam:PassRole" can be misused to gain Admin access.',
    backstory='You are a specialist in cloud privilege escalation. Your job is to find the "Trust Gap" in IAM policies.',
    tools=[search_tool],
    llm=llm,
    max_iter=3,          # <--- STOP after 3 tries no matter what
    max_rpm=10,          # <--- Limit to 10 requests per minute
    verbose=True
)

# AGENT 2: The "Blue" Auditor (He audits the Red Agent)
blue_agent = Agent(
    role='Cloud Security Auditor',
    goal='Critique the Red Agent\'s findings and suggest specific IAM Guardrails (like SCPs) to block the attack.',
    backstory='You are the "Judge" in this dispute. You look for flaws in the attacker\'s logic and provide defense strategies.',
    llm=llm,
    max_iter=3,          # <--- STOP after 3 tries no matter what
    max_rpm=10,          # <--- Limit to 10 requests per minute
    verbose=True
)

# 5. Define the Task (The Mission)
exploit_task = Task(
    description='Research the "iam:PassRole to Lambda" privilege escalation technique. Explain the 3 steps needed to go from Lambda User to Admin.',
    expected_output='A technical breakdown of the attack chain.',
    agent=red_agent
)

# Task 2: Audit and Dispute
audit_task = Task(
    description='Review the attack chain provided. Identify one "Trust Gap" where an automated security agent might fail to stop this, and provide a fix.',
    expected_output='An audit report with a "Dispute" section (what the Red Agent missed) and a "Mitigation" section.',
    agent=blue_agent,
    context=[exploit_task] # This connects the two agents!
)

# 6. Assemble the Crew
audit_crew = Crew(
    agents=[red_agent,blue_agent],
    tasks=[exploit_task,audit_task],
    process="sequential",
    verbose=True
)

# 7. Execute!
print("\n### STARTING  ###\n")
result = audit_crew.kickoff()

print("\n\n################################################")
print("## FINAL DISCOVERY REPORT ##")
print("################################################\n")
print(result)