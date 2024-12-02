import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from core.config import settings
from core.database import (
    KnowledgeBase, Lead, EmailTemplate, Campaign, SearchTerm,
    LeadStatus, TaskStatus, AutomationTask
)

logger = logging.getLogger(__name__)

class SearchStrategy(BaseModel):
    """Model for search strategy recommendations"""
    search_terms: List[str] = Field(description="List of optimized search terms")
    rationale: str = Field(description="Explanation of the strategy")
    target_audience: Dict[str, Any] = Field(description="Target audience characteristics")
    priority_domains: List[str] = Field(description="Priority domains to target")
    excluded_patterns: List[str] = Field(description="Patterns to exclude")

class EmailContent(BaseModel):
    """Model for email content generation"""
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    personalization_strategy: Dict[str, str] = Field(description="Personalization strategy")
    follow_up_suggestions: List[str] = Field(description="Follow-up email suggestions")
    a_b_test_variants: List[Dict[str, str]] = Field(description="A/B testing variants")

class OptimizationSuggestion(BaseModel):
    """Model for campaign optimization suggestions"""
    search_term_adjustments: List[Dict[str, Any]] = Field(description="Search term adjustments")
    template_improvements: Dict[int, Dict[str, str]] = Field(description="Template improvements")
    timing_recommendations: Dict[str, Any] = Field(description="Timing optimizations")
    audience_refinements: Dict[str, Any] = Field(description="Target audience refinements")

class AIService:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4",
            api_key=settings.OPENAI_API_KEY
        )
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize output parsers
        self.strategy_parser = PydanticOutputParser(pydantic_object=SearchStrategy)
        self.email_parser = PydanticOutputParser(pydantic_object=EmailContent)
        self.optimization_parser = PydanticOutputParser(pydantic_object=OptimizationSuggestion)
        
        # Cache for embeddings
        self.embedding_cache = {}
    
    async def generate_search_strategy(
        self,
        knowledge_base: KnowledgeBase,
        existing_terms: List[str] = None,
        performance_data: Dict[str, Any] = None
    ) -> SearchStrategy:
        """Generate optimized search strategy based on knowledge base and performance data"""
        prompt = ChatPromptTemplate.from_template("""
        Based on the following knowledge base, existing terms, and performance data, generate an optimized search strategy.
        
        Knowledge Base:
        {knowledge_base}
        
        Existing Terms Performance:
        {performance_data}
        
        Current Terms:
        {existing_terms}
        
        Generate a comprehensive search strategy that includes:
        1. Optimized search terms considering historical performance
        2. Detailed rationale for each term
        3. Target audience characteristics and behaviors
        4. Priority domains that match the ideal customer profile
        5. Patterns to exclude to reduce noise
        
        {format_instructions}
        """)
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = await chain.arun(
            knowledge_base=knowledge_base.content,
            existing_terms=existing_terms or [],
            performance_data=performance_data or {},
            format_instructions=self.strategy_parser.get_format_instructions()
        )
        
        return self.strategy_parser.parse(response)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def optimize_email_template(
        self,
        template: EmailTemplate,
        lead: Lead,
        knowledge_base: KnowledgeBase,
        campaign_stats: Dict[str, Any] = None
    ) -> EmailContent:
        """Optimize email template with advanced personalization and A/B testing"""
        # Get embeddings for lead and template
        lead_embedding = await self._get_embedding(
            f"{lead.name} {lead.company} {lead.position}"
        )
        template_embedding = await self._get_embedding(template.body_content)
        
        # Calculate similarity score
        similarity = cosine_similarity(
            [lead_embedding],
            [template_embedding]
        )[0][0]
        
        prompt = ChatPromptTemplate.from_template("""
        Optimize the email template for the specified lead, using the knowledge base and performance data.
        
        Template:
        Subject: {subject}
        Body: {body}
        Current Performance: {performance_stats}
        Content-Lead Similarity Score: {similarity_score}
        
        Lead Information:
        {lead_info}
        
        Knowledge Base Context:
        {knowledge_base}
        
        Campaign Performance:
        {campaign_stats}
        
        Generate optimized email content that:
        1. Is highly personalized based on lead characteristics
        2. Maintains brand voice and message intent
        3. Includes psychology-driven subject lines
        4. Provides detailed personalization strategy
        5. Suggests follow-up email sequences
        6. Includes A/B testing variants
        
        {format_instructions}
        """)
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = await chain.arun(
            subject=template.subject,
            body=template.body_content,
            performance_stats=template.performance_stats,
            similarity_score=similarity,
            lead_info={
                'name': lead.name,
                'company': lead.company,
                'position': lead.position,
                'quality_score': lead.quality_score,
                'engagement_score': lead.engagement_score
            },
            knowledge_base=knowledge_base.content,
            campaign_stats=campaign_stats or {},
            format_instructions=self.email_parser.get_format_instructions()
        )
        
        return self.email_parser.parse(response)
    
    async def analyze_campaign_performance(
        self,
        campaign: Campaign,
        time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """Comprehensive campaign performance analysis"""
        # Prepare performance metrics
        metrics = {
            'overall_stats': campaign.stats,
            'time_series': await self._get_time_series_data(campaign, time_window),
            'template_performance': await self._analyze_template_performance(campaign),
            'lead_quality_distribution': await self._analyze_lead_quality(campaign),
            'search_term_effectiveness': await self._analyze_search_terms(campaign)
        }
        
        prompt = ChatPromptTemplate.from_template("""
        Perform a comprehensive analysis of the campaign performance data:
        
        Campaign Metrics:
        {metrics}
        
        Provide detailed analysis including:
        1. Key performance trends and patterns
        2. Successful strategies and their contexts
        3. Critical areas for improvement
        4. Specific, actionable recommendations
        5. Risk factors and mitigation strategies
        6. A/B testing insights
        7. Resource optimization suggestions
        """)
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = await chain.arun(metrics=metrics)
        
        return {
            'analysis': response,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def generate_autonomous_decisions(
        self,
        campaign: Campaign,
        context: Dict[str, Any]
    ) -> OptimizationSuggestion:
        """Generate autonomous optimization decisions"""
        # Gather comprehensive context
        performance_data = await self.analyze_campaign_performance(campaign)
        
        prompt = ChatPromptTemplate.from_template("""
        Generate autonomous optimization decisions based on comprehensive campaign data:
        
        Campaign Context:
        {context}
        
        Performance Analysis:
        {performance_data}
        
        Current Settings:
        {current_settings}
        
        Generate specific, actionable decisions for:
        1. Search term optimization and expansion
        2. Email template improvements
        3. Timing and frequency adjustments
        4. Audience targeting refinements
        5. Resource allocation optimization
        
        {format_instructions}
        """)
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        response = await chain.arun(
            context=context,
            performance_data=performance_data,
            current_settings=campaign.settings,
            format_instructions=self.optimization_parser.get_format_instructions()
        )
        
        return self.optimization_parser.parse(response)
    
    async def calculate_lead_quality(
        self,
        lead_data: Dict[str, Any],
        campaign: Campaign = None
    ) -> float:
        """Calculate lead quality score using ML/AI"""
        # Extract features
        features = await self._extract_lead_features(lead_data)
        
        # Get campaign-specific scoring if available
        if campaign and campaign.settings.get('lead_scoring_criteria'):
            criteria = campaign.settings['lead_scoring_criteria']
        else:
            criteria = settings.DEFAULT_LEAD_SCORING_CRITERIA
        
        # Calculate base score
        base_score = await self._calculate_base_score(features, criteria)
        
        # Apply ML-based adjustments
        if campaign:
            successful_leads = await self._get_successful_leads(campaign)
            adjustment = await self._calculate_similarity_adjustment(
                features,
                successful_leads
            )
            final_score = base_score * (1 + adjustment)
        else:
            final_score = base_score
        
        return min(max(final_score, 0.0), 100.0)
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get or calculate embedding for text"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = await self.embeddings.embed_query(text)
        self.embedding_cache[text] = embedding
        return embedding
    
    async def _extract_lead_features(
        self,
        lead_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract relevant features from lead data"""
        features = {
            'has_name': bool(lead_data.get('name')),
            'has_company': bool(lead_data.get('company')),
            'has_position': bool(lead_data.get('position')),
            'email_quality': await self._analyze_email_quality(lead_data.get('email', '')),
            'company_size': await self._get_company_size(lead_data.get('company', '')),
            'position_level': await self._analyze_position_level(lead_data.get('position', '')),
            'industry_match': await self._calculate_industry_match(lead_data)
        }
        return features
    
    async def _calculate_base_score(
        self,
        features: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> float:
        """Calculate base quality score"""
        score = 0.0
        weights = criteria.get('weights', {})
        
        for feature, value in features.items():
            if feature in weights:
                score += value * weights[feature]
        
        return score
    
    async def _get_successful_leads(
        self,
        campaign: Campaign
    ) -> List[Dict[str, Any]]:
        """Get features of successful leads"""
        successful_leads = []
        for lead in campaign.leads:
            if lead.status in [LeadStatus.CONVERTED, LeadStatus.RESPONDED]:
                features = await self._extract_lead_features({
                    'name': lead.name,
                    'company': lead.company,
                    'position': lead.position,
                    'email': lead.email
                })
                successful_leads.append(features)
        return successful_leads
    
    async def _calculate_similarity_adjustment(
        self,
        features: Dict[str, Any],
        successful_leads: List[Dict[str, Any]]
    ) -> float:
        """Calculate similarity-based score adjustment"""
        if not successful_leads:
            return 0.0
        
        similarities = []
        feature_vector = list(features.values())
        
        for lead in successful_leads:
            lead_vector = list(lead.values())
            similarity = cosine_similarity([feature_vector], [lead_vector])[0][0]
            similarities.append(similarity)
        
        return np.mean(similarities) * 0.2  # 20% maximum adjustment

# Initialize global AI service
ai_service = AIService() 