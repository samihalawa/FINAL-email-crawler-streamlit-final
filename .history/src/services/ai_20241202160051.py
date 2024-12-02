import logging
from typing import List, Dict, Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from datetime import datetime

from core.database import Lead, KnowledgeBase

class AIService:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        self.memory = ConversationBufferMemory()
        
    async def optimize_search_terms(self, base_terms: List[str], kb_info: Dict) -> List[str]:
        """Optimize search terms using AI"""
        try:
            prompt = ChatPromptTemplate.from_template("""
            Given these base search terms: {base_terms}
            And this knowledge base information about the business:
            {kb_info}
            
            Generate optimized search terms that will find potential clients while following these rules:
            1. Avoid competitor keywords (e.g., "best AI courses", "automation solutions")
            2. Focus on client descriptors (industry, services, titles, locations)
            3. Exclude AI-specific terms
            4. Use geographic and industry filters
            5. Ensure terms will lead to actual contact information
            
            Return only the list of optimized search terms, one per line.
            """)
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            result = await chain.arun({
                'base_terms': '\n'.join(base_terms),
                'kb_info': str(kb_info)
            })
            
            # Process result into list
            optimized_terms = [
                term.strip()
                for term in result.split('\n')
                if term.strip()
            ]
            
            return optimized_terms
            
        except Exception as e:
            logging.error(f"Error optimizing search terms: {str(e)}")
            return base_terms
            
    async def personalize_email(self, template: Dict, lead: Lead) -> Dict:
        """Personalize email template for specific lead"""
        try:
            prompt = ChatPromptTemplate.from_template("""
            Given this email template:
            Subject: {subject}
            Body: {body}
            
            And this lead information:
            Email: {email}
            Name: {name}
            Company: {company}
            Source URL: {source_url}
            
            Personalize the email while:
            1. Maintaining the original message intent
            2. Adding natural, contextual personalization
            3. Keeping a professional tone
            4. Making it feel genuine and non-automated
            
            Return the personalized subject and body.
            Format: SUBJECT: <subject>\nBODY: <body>
            """)
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            result = await chain.arun({
                'subject': template['subject'],
                'body': template['body'],
                'email': lead.email,
                'name': lead.name or '',
                'company': lead.company or '',
                'source_url': lead.source_url or ''
            })
            
            # Parse result
            parts = result.split('\n', 1)
            subject = parts[0].replace('SUBJECT:', '').strip()
            body = parts[1].replace('BODY:', '').strip()
            
            return {
                'subject': subject,
                'body': body,
                'personalization_strategy': {
                    'name_used': bool(lead.name),
                    'company_used': bool(lead.company),
                    'source_referenced': bool(lead.source_url)
                }
            }
            
        except Exception as e:
            logging.error(f"Error personalizing email: {str(e)}")
            return {
                'subject': template['subject'],
                'body': template['body'],
                'personalization_strategy': {}
            }
            
    async def analyze_response(self, response_text: str) -> Dict:
        """Analyze email response for sentiment and intent"""
        try:
            prompt = ChatPromptTemplate.from_template("""
            Analyze this email response:
            {response_text}
            
            Determine:
            1. Overall sentiment (positive, neutral, negative)
            2. Level of interest (high, medium, low)
            3. Key objections or concerns
            4. Suggested follow-up approach
            
            Format your response as:
            SENTIMENT: <sentiment>
            INTEREST: <interest>
            OBJECTIONS: <objections>
            FOLLOW_UP: <approach>
            """)
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            result = await chain.arun({
                'response_text': response_text
            })
            
            # Parse result into structured data
            lines = result.split('\n')
            analysis = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    analysis[key.strip().lower()] = value.strip()
                    
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing response: {str(e)}")
            return {
                'sentiment': 'neutral',
                'interest': 'unknown',
                'objections': '',
                'follow_up': ''
            }
            
    async def generate_follow_up(self, lead: Lead, previous_emails: List[Dict], kb_info: Dict) -> Dict:
        """Generate follow-up email based on conversation history"""
        try:
            prompt = ChatPromptTemplate.from_template("""
            Given this conversation history:
            {conversation_history}
            
            And this lead information:
            {lead_info}
            
            And this knowledge base:
            {kb_info}
            
            Generate a follow-up email that:
            1. References previous interactions
            2. Addresses any concerns or objections
            3. Provides relevant value proposition
            4. Includes a clear call to action
            
            Format: SUBJECT: <subject>\nBODY: <body>
            """)
            
            # Format conversation history
            history = '\n\n'.join([
                f"From: {email['from']}\nTo: {email['to']}\nDate: {email['date']}\n{email['content']}"
                for email in previous_emails
            ])
            
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory
            )
            
            result = await chain.arun({
                'conversation_history': history,
                'lead_info': str(lead),
                'kb_info': str(kb_info)
            })
            
            # Parse result
            parts = result.split('\n', 1)
            subject = parts[0].replace('SUBJECT:', '').strip()
            body = parts[1].replace('BODY:', '').strip()
            
            return {
                'subject': subject,
                'body': body
            }
            
        except Exception as e:
            logging.error(f"Error generating follow-up: {str(e)}")
            return None 