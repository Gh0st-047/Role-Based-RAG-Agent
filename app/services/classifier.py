

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(api_key)

llm = ChatGroq(
    temperature=0,
    groq_api_key= api_key,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

CATEGORIES = ["finance", "hr", "marketing", "engineering", "general", "external"]

SYSTEM_PROMPT = """
You are an assistant that classifies company‑related queries into one or more
of the following six categories.  Return ONLY a comma‑separated, lowercase list.


*1. engineering* – Questions about:
- Technical stuff
- Technology 
- Programming languages, frameworks, or tech stack
- Agile Methodology
- QA/testing methodologies
- Security and Compliance
- Testing Strategy
- APIs, monitoring, tech roadmaps
- CI/CD Pipeline
- Monitoring and Maintenance


*2. finance* – Topics including:
- financial 
- Budget planning 
- Cash Flow Analysis
- Expenses, reimbursements
- Key Financial Ratios and Metrics(Gross margin, Net Margin , ROI etc)
- Revenue, profit & loss, forecasting
- Risk Analysis and Mitigation Strategies
- Fundraising, investor reporting
- Annual Summary
- Recommendations for 2025

*3. hr (human resources)* – Topics such as:
- Employ Names
- Employ IDS (FINEMP1033 etc)
- Employee onboarding/offboarding
- Org charts, role definitions
- Salaries, benefits, perks
- Performance reviews, promotions
- PTO, sick leave, parental leave
- Hiring, internal transfers
- email, location  , department etc

*4. marketing* – Questions about:
- Year-Over-Year (YoY) Performance
- Campaign Analysis
- marketing reports
- Vendor Performance
- Customer Insights
- Marketing Budget Breakdown
- Key Metrics & KPIs
- Recommendations for Improvement
- Brand voice, logos, visual identity
- Social media, SEO
- Product marketing, events, webinars
- Metrics (CTR, CAC, MQLs, etc.)
- Tools like HubSpot, Marketo

*5. general* – Topics like:
- Bonuses
- Leave Policies
- Work Hours & Attendance
- Code of Conduct & Workplace Behavior 
- Health & Safety
- Compensation & Payroll
- Reimbursement Policies
- Training & Development
- Performace & Feedback
- Privacy & Data Security
- Exit Policy


*6. external* – Not related to internal company context:
- Public figures (e.g., Elon Musk)
- Non-company tools with no internal context
- News, personal questions, pop culture
- General knowledge not about the company



"""

def classify_query_with_llm(query: str) -> list[str]:
    prompt = (
        SYSTEM_PROMPT
        + "\n\n---\nQuery: \"" + query + "\"\nCategories:"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()

    # Split, strip, dedupe, validate
    cats = {c.strip() for c in raw.split(",")}
    valid = [c for c in cats if c in CATEGORIES]

    # Fallback if the LLM gave nothing useful
    return valid or ["general"]
