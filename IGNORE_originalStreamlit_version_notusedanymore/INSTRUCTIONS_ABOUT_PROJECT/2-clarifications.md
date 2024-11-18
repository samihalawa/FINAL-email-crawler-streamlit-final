Important Clarifications & Requirements Verification

1. Manual Long-Running Processes
- Bulk send functionality runs as background process until completion
- Manual search functionality runs as background process until completion 
- These processes must:
  * Continue running until target lead count reached
  * Utilize HuggingFace Space for Python execution
  * Be orchestrated by Cloudflare Worker
  * Maintain state and progress
  * Resume after interruptions
  * Provide real-time status updates

2. AI Search Term Enhancement
The AI enhancement prompt must be:
"You are an AI marketing automation tool expert. Your role is to generate more optimized search terms related to the following search term group, so all can be scraped together and contacted together with the same email marketing and make sense and actually are optimized according to what you think the objective of the user is from the search term group name and the already existing search terms related etc"

3. Group Management
- Search term groups must be created manually
- Groups are managed through search term views interface in streamlit_app.py
- No automated group creation/management

4. Processing Requirements
A term is considered "processed" only when both:
- Leads are successfully found through search
- Emails are successfully sent to those leads
This matches manual_search_page with email=ON

5. Unprocessed Terms Handling
For terms not processed in current loop:
- Terms remain pending for next loop iteration
- Unprocessed terms receive priority in subsequent loops

6. Distribution Method
Equitable distribution algorithm must:
- Calculate equal email quota per search term
- Process terms homogeneously across groups
- Balance coverage within max_emails_per_group limit

7. UI/UX Requirements
Automation page must implement:
- Dynamic group processing visualization
- Smooth group collapse/expand animations  
- Search term highlighting/unhighlighting
- Real-time auto-scrolling logs displaying:
  * Search progress
  * Lead discovery
  * Email sending status
  * Detailed process updates