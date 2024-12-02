# Autoclient Project Requirements and Criteria

## Core Objectives
- Build an effective email crawler and lead generation system
- Focus on finding decision-makers across various industries
- Generate high-quality leads beyond just marketing companies

## What We Want

### Data Sources
- Business directory websites
- Company listing platforms
- Professional networks
- Industry-specific forums
- Chamber of commerce listings
- Local business associations
- Company websites
- Professional social networks
- Industry event websites
- Regional business registries

### Target Information
- Company email addresses
- Decision maker contact details
- Company size and location
- Industry sector
- Company description
- Website URLs
- Social media profiles
- Phone numbers (when available)

### Search Approach
- Geographic-based searches (e.g., "empresas barcelona", "negocios madrid")
- Industry-specific directories
- Business type searches without AI/automation terms
- Local business listings
- Professional associations
- Industry clusters
- Business networks

### Email Collection Strategy
- Direct from company websites
- Public business directories
- Professional networks
- Contact pages
- Team/About pages
- Footer sections
- Press/Media sections

## What We Don't Want

### Search Terms to Avoid
- AI-related keywords
- Marketing agency focused terms
- Automation-related terms
- Technology adoption terms
- Innovation-related keywords
- Digital transformation terms

### Data Sources to Avoid
- Personal email addresses
- Generic contact forms
- Support/info@ addresses
- Non-business email domains
- Competitor websites
- Training/course providers
- Technology vendor lists

### Approaches to Avoid
- Assuming Google search is the only source
- Limiting to marketing companies only
- Using technical/AI terminology
- Focusing only on large companies
- Targeting only tech-savvy businesses
- Relying on single-source data

## Technical Requirements

### Crawler Behavior
- Respect robots.txt
- Handle rate limiting
- Verify email validity
- Remove duplicates
- Filter spam traps
- Handle different page structures
- Support multiple languages
- Process pagination
- Handle redirects
- Manage session cookies

### Data Processing
- Email format validation
- Remove generic emails
- Deduplicate entries
- Standardize data format
- Clean contact information
- Verify company details
- Format consistency
- Data categorization
- Quality scoring
- Source tracking

### Output Format
- Structured JSON/CSV
- Complete contact details
- Source attribution
- Timestamp collection
- Confidence scores
- Verification status
- Category tags
- Company metadata
- Contact role/position
- Data quality indicators 