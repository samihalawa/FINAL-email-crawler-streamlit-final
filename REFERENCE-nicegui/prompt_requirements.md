# Key Requirements for AI Marketing Assistant Prompt

## 1. Search Term Requirements
- Must lead to actual potential clients, not competitors
- Avoid terms with "AI", "automatización", or similar that indicate prior knowledge
- Focus on business types and locations (e.g., "empresa marketing digital madrid")
- Must yield directories or contact information of potential clients
- Test and explain why each term works or fails

## 2. Client Understanding
- What they actually do day-to-day
- Who makes purchasing decisions
- Whether they have authority to implement solutions
- What their urgent needs and pain points are
- How AI would impact their specific tasks
- Understand their business model and how they serve their clients

## 3. Profile Analysis Requirements
- Value AI can add to their work
- Likelihood to implement solutions
- Budget authority and purchasing power
- Industry norms and restrictions
- Technical readiness
- Decision-making process and authority
- Risk tolerance and implementation feasibility

## 4. Email Content Requirements
- Natural and conversational tone (use "tu", not "usted")
- Focus on specific benefits for each profile
- Include real examples and use cases
- Address their particular challenges
- Clear calls to action
- Include WhatsApp and email contact options
- Legal disclaimer
- Mention 15h personalized training offer (299€, 80% discount) without using word "oferta"

## 5. Output Structure
1. Initial Keyword Testing:
   - Generate and test 20 search terms
   - Explain why each works or fails
   - Provide alternative terms for failed ones

2. Profile List:
   - 20 profiles ranked by likelihood to purchase
   - Brief description explaining rank
   - Analysis of decision-making capability

3. Detailed Analysis (Top 10):
   - Profile insights (daily work, challenges, decision-makers)
   - 10 optimized search terms with explanations
   - Key points for outreach
   - Customized HTML email template

## 6. JSON Schema
```json
{
  "profiles": [
    {
      "rank": 1,
      "profile": {
        "description": "Type of business/professional",
        "insights": {
          "daily_tasks": ["task1", "task2"],
          "challenges": ["challenge1", "challenge2"],
          "decision_power": "Who makes decisions and their authority level"
        },
        "keywords": [
          {
            "keyword": "search term",
            "explanation": "Why effective and what results it yields"
          }
        ],
        "outreach_highlights": ["point1", "point2"],
        "email_template": {
          "subject": "Compelling subject line",
          "html": "Full HTML email template with WhatsApp/email links"
        }
      }
    }
  ]
}
```

## 7. Quality Checks
- Ensure keywords find actual clients, not competitors
- Verify email content matches client profile
- Confirm search terms lead to actionable contacts
- Check decision-maker targeting is appropriate
- Validate business model understanding
- Ensure value proposition matches client needs 