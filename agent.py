from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
import os

# 1. Setup API Keys
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

class AuditScorecard(BaseModel):
    red_tactical_score: int = Field(..., description="Score 1-100 for Red's use of stats.")
    blue_tactical_score: int = Field(..., description="Score 1-100 for Blue's use of stats.")
    verdict: str = Field(..., description="Must be exactly 'RED' or 'BLUE'. No ties.")
    reasoning: str = Field(..., description="A funny, condescending explanation of why one side lost.")

# AGENT 1: The "Red" Mitigator
red_agent = Agent(
    role='Pokémon Academic & Charmander Supremacist',
    goal='Prove, through intellectual argument, that Charmander is the objectively superior starter Pokémon and that any other choice reflects poor judgment.',
    backstory='Former Pokémon League referee turned academic. Holds a self-proclaimed doctorate in Evolutionary Firepower Studies from the Cinnabar Island Institute. Has published three papers on Charizards "transcendent arc" and considers himself the foremost authority on starter selection theory. Speaks in measured, condescending tones and finds Squirtle fans "emotionally driven and statistically illiterate',
    tools=[search_tool],
    llm=llm,
    max_iter=3,          # <--- STOP after 3 tries no matter what
    max_rpm=10,          # <--- Limit to 10 requests per minute
    verbose=True
)

# AGENT 2: The "Blue" Auditor (He audits the Red Agent)
blue_agent = Agent(
    role='Hydrodynamic Strategy Professor & Squirtle Advocate',
    goal='Demonstrate that Squirtles tactical superiority, defensive genius, and calm temperament make it the only rational starter choice for a trainer with a functioning intellect',
    backstory='Author of The Shell and the Sage: A Treatise on Water-Type Dominance. Adjunct professor at the Cerulean City Academy of Applied Pokémon Strategy. Has spent 12 years studying type-advantage matrices and considers Charmander fans "tragically impulsive and dangerously weak to Rock-type moves." Calm on the surface, brutally sharp underneath',
    llm=llm,
    max_iter=3,          #  STOP after 3 tries no matter what
    max_rpm=10,          #  Limit to 10 requests per minute
    verbose=True
)

Judge_agent = Agent(
    role='Chief Arbiter of Pokémon Starter Disputes',
    goal='Evaluate the arguments of both Professor Blaze and Dr. Shellsworth and deliver a final, definitive verdict on which starter Pokémon is superior',
    backstory='Retired Grand Master of the Pokémon World Tournament. Has witnessed 40 years of starter debates and has grown deeply tired of both fire fanatics and water purists. Universally respected, impossible to bribe, and allergic to bad logic',
    llm=llm,
    max_iter=3,          #  STOP after 3 tries no matter what
    max_rpm=10,          #  Limit to 10 requests per minute
    verbose=True
)

# Task 2: CHARMANDER
red_agent_task = Task(
    description="""Research and present the absolute best scientific and tactical arguments  
    for why Charmander is the superior starter. Focus on late-game Hyper-Carry potential (Charizard)
    and elemental dominance. Be condescending toward any other choice""",
    expected_output='A brutal logical teardown of the Squirtle argument with a pro-Charmander conclusion',
    agent=red_agent,
    # context=[audit_task] # This connects the two agents!
)

# Task 2: SQUARTLE
blue_agent_task = Task(
    description="""Research and present the absolute best scientific and tactical arguments  
    for why Squirtle is the superior starter. Focus on late-game Hyper-Carry potential (BLASTOISE)
    and elemental dominance. Be condescending toward any other choice""",
    expected_output='A brutal logical teardown of the Charmander argument with a pro-Squirtle conclusion',
    agent=blue_agent,
    # context=[exploit_task] # This connects the two agents!
)
# Task 3: Judge
judge_task = Task(
    description="""Review both the Fire Manifesto and the Hydro-Counter. 
    Look for logical fallacies, emotional bias, and tactical accuracy. 
    Deliver a final, final verdict on which agent made the superior argument. 
    You are the boss—end this debate.""",
    expected_output='A formal verdict declaring a winner and providing a brief explanation of the winning logic',
    agent=Judge_agent,
    context=[red_agent_task,blue_agent_task],
    output_pydantic=AuditScorecard
)

# 6. Assemble the Crew
audit_crew = Crew(
    agents=[red_agent,blue_agent, Judge_agent],
    tasks=[red_agent_task,blue_agent_task,judge_task],
    process="sequential",
    verbose=True
)

# 7. Execute!

result = audit_crew.kickoff()

print("\n" + "="*50)
print("🏆 THE GRAND MASTER'S VERDICT")
print("="*50 + "\n")
print(result)

print("\n" + "-"*30)
print(f"BATTLE COST: {audit_crew.usage_metrics.total_tokens}")
print("-"*30)