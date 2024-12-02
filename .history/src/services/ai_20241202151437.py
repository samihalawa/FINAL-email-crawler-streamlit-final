import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import settings
from core.database import KnowledgeBase, Lead, EmailTemplate

logger = logging.getLogger(__name__)

class SearchStrategy(BaseModel):
    """Model for search strategy recommendations"""
    search_terms: List[str] = Field(description="List of optimized search terms")
    rationale: str = Field(description="Explanation of the strategy")
    target_audience: Dict[str, Any] = Field(description="Target audience characteristics")

class EmailContent(BaseModel):
    """Model for email content generation"""
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    personalization_strategy: Dict[str, str] = Field(description="Personalization strategy")

class AIService:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4",
            api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize output parsers
        self.strategy_parser = PydanticOutputParser(pydantic_object=SearchStrategy)
        self.email_parser = PydanticOutputParser(pydantic_object=EmailContent)
    
    async def generate_search_strategy(
        self,
        knowledge_base: KnowledgeBase,
        existing_terms: List[str] = None
    ) -> SearchStrategy:
        """Generate optimized search strategy based on knowledge base"""
        prompt = ChatPromptTemplate.from_template("""
        Based on the following knowledge base and existing search terms, generate an optimized search strategy.
        
        Knowledge Base:
        {knowledge_base}
        
        Existing Terms:
        {existing_terms}
        
        Generate a search strategy that includes:
        1. Optimized search terms
        2. Rationale for the strategy
        3. Target audience characteristics
        
        {format_instructions}
        """)
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = await chain.arun(
            knowledge_base=knowledge_base.content,
            existing_terms=existing_terms or [],
            format_instructions=self.strategy_parser.get_format_instructions()
        )
        
        return self.strategy_parser.parse(response)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def optimize_email_template(
        self,
        template: EmailTemplate,
        lead: Lead,
        knowledge_base: KnowledgeBase
    ) -> EmailContent:
        """Optimize email template for specific lead"""
        prompt = ChatPromptTemplate.from_template("""
        Optimize the following email template for the specified lead, using the knowledge base for context.
        
        Template:
        Subject: {subject}
        Body: {body}
        
        Lead Information:
        - Name: {lead_name}
        - Company: {lead_company}
        - Position: {lead_position}
        
        Knowledge Base Context:
        {knowledge_base}
        
        Generate optimized email content that:
        1. Is highly personalized
        2. Maintains the original message intent
        3. Includes effective subject line
        4. Provides personalization strategy
        
        {format_instructions}
        """)
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = await chain.arun(
            subject=template.subject,
            body=template.body_content,
            lead_name=lead.name,
            lead_company=lead.company,
            lead_position=lead.position,
            knowledge_base=knowledge_base.content,
            format_instructions=self.email_parser.get_format_instructions()
        )
        
        return self.email_parser.parse(response)
    
    async def analyze_campaign_performance(
        self,
        campaign_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze campaign performance and provide recommendations"""
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following campaign performance data and provide recommendations:
        
        Campaign Data:
        {campaign_data}
        
        Provide analysis including:
        1. Key performance indicators
        2. Successful patterns
        3. Areas for improvement
        4. Specific recommendations
        """)
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = await chain.arun(campaign_data=campaign_data)
        
        return {
            'analysis': response,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def generate_autonomous_decisions(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate autonomous decisions for campaign optimization"""
        prompt = ChatPromptTemplate.from_template("""
        Based on the following context, generate autonomous decisions for campaign optimization:
        
        Context:
        {context}
        
        Generate decisions regarding:
        1. Search term adjustments
        2. Email content modifications
        3. Timing optimizations
        4. Target audience refinements
        """)
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = await chain.arun(context=context)
        
        return {
            'decisions': response,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def validate_lead_quality(
        self,
        lead: Lead,
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate lead quality against specified criteria"""
        prompt = ChatPromptTemplate.from_template("""
        Evaluate the following lead against the specified criteria:
        
        Lead Information:
        {lead_info}
        
        Evaluation Criteria:
        {criteria}
        
        Provide:
        1. Quality score (0-100)
        2. Match confidence
        3. Specific matches/mismatches
        4. Recommendations
        """)
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = await chain.arun(
            lead_info={
                'name': lead.name,
                'company': lead.company,
                'position': lead.position,
                'email': lead.email
            },
            criteria=criteria
        )
        
        return {
            'evaluation': response,
            'timestamp': datetime.utcnow().isoformat()
        } 