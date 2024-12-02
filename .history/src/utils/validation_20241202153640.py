from typing import List, Dict, Any, Optional
import re
from email.utils import parseaddr
from urllib.parse import urlparse
import tld
from datetime import datetime

from core.logging import ValidationError

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        if not email or '@' not in email:
            return False
        
        name, addr = parseaddr(email)
        if not addr:
            return False
        
        try:
            local, domain = addr.split('@')
            return (
                re.match(r'^[a-zA-Z0-9._%+-]+$', local) and
                re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', domain)
            )
        except:
            return False
    
    @staticmethod
    def validate_domain(domain: str) -> bool:
        """Validate domain format"""
        try:
            result = urlparse(f"http://{domain}")
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format"""
        # Basic phone validation - can be enhanced based on requirements
        return bool(re.match(r'^\+?1?\d{9,15}$', phone))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def sanitize_html(html: str) -> str:
        """Remove potentially dangerous HTML"""
        # This is a basic implementation - consider using a proper HTML sanitizer library
        return re.sub(r'<[^>]*?>', '', html)
    
    @staticmethod
    def validate_date(date_str: str, format: str = '%Y-%m-%d') -> bool:
        """Validate date string format"""
        try:
            datetime.strptime(date_str, format)
            return True
        except:
            return False

class InputSanitizer:
    """Input sanitization utilities"""
    
    @staticmethod
    def clean_email(email: str) -> str:
        """Clean and normalize email address"""
        email = email.strip().lower()
        name, addr = parseaddr(email)
        return addr
    
    @staticmethod
    def clean_domain(domain: str) -> str:
        """Clean and normalize domain"""
        domain = domain.strip().lower()
        if domain.startswith(('http://', 'https://')):
            domain = urlparse(domain).netloc
        return domain.split(':')[0]  # Remove port if present
    
    @staticmethod
    def clean_phone(phone: str) -> str:
        """Clean and normalize phone number"""
        return re.sub(r'[^\d+]', '', phone)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text input"""
        return text.strip()

class LeadValidator:
    """Lead data validation"""
    
    @staticmethod
    def validate_lead(data: Dict[str, Any]) -> List[str]:
        """Validate lead data"""
        errors = []
        
        # Required fields
        required_fields = ['email', 'first_name', 'company']
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Email validation
        if data.get('email'):
            if not DataValidator.validate_email(data['email']):
                errors.append("Invalid email format")
        
        # Website validation
        if data.get('website'):
            if not DataValidator.validate_url(data['website']):
                errors.append("Invalid website URL")
        
        # Phone validation
        if data.get('phone'):
            if not DataValidator.validate_phone(data['phone']):
                errors.append("Invalid phone number")
        
        return errors

class SearchValidator:
    """Search query validation"""
    
    @staticmethod
    def validate_search_terms(terms: List[str]) -> List[str]:
        """Validate search terms"""
        errors = []
        
        if not terms:
            errors.append("No search terms provided")
        
        for term in terms:
            if len(term.strip()) < 3:
                errors.append(f"Search term too short: {term}")
            if len(term) > 100:
                errors.append(f"Search term too long: {term}")
        
        return errors
    
    @staticmethod
    def validate_excluded_domains(domains: List[str]) -> List[str]:
        """Validate excluded domains"""
        errors = []
        
        for domain in domains:
            if not DataValidator.validate_domain(domain):
                errors.append(f"Invalid domain: {domain}")
        
        return errors

class EmailValidator:
    """Email content validation"""
    
    @staticmethod
    def validate_template(template: Dict[str, Any]) -> List[str]:
        """Validate email template"""
        errors = []
        
        # Required fields
        if not template.get('subject'):
            errors.append("Missing email subject")
        if not template.get('body'):
            errors.append("Missing email body")
        
        # Length limits
        if len(template.get('subject', '')) > 100:
            errors.append("Subject line too long (max 100 chars)")
        
        # Check for required placeholders
        required_placeholders = ['{first_name}', '{company}']
        for placeholder in required_placeholders:
            if placeholder not in template.get('body', ''):
                errors.append(f"Missing required placeholder: {placeholder}")
        
        return errors
    
    @staticmethod
    def validate_sender(sender: str) -> List[str]:
        """Validate sender email"""
        errors = []
        
        if not DataValidator.validate_email(sender):
            errors.append("Invalid sender email format")
        
        # Additional sender validation (e.g., domain verification) can be added here
        
        return errors

def validate_or_raise(validation_func, data, error_prefix: str = "Validation error"):
    """Utility function to validate data and raise exception if invalid"""
    errors = validation_func(data)
    if errors:
        raise ValidationError(f"{error_prefix}: {'; '.join(errors)}")
    return True 