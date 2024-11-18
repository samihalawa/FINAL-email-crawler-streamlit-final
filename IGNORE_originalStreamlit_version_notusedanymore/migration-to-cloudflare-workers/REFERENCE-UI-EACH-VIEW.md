# Autoclient.ai Pages Reference

Detailed page layouts and functionality required for each section:

## 1. Manual Search (🔍)
- Search terms input with tags
- Results per term slider (1-500)
- Configuration options:
  - Enable email sending toggle
  - Ignore fetched domains toggle
  - Shuffle keywords toggle
  - Optimize (English) toggle
  - Optimize (Spanish) toggle
  - Language selector (ES/EN)
- Email sending configuration:
  - Template selector
  - From email selector
  - Reply-to field
- Search progress indicators:
  - Progress bar
  - Status text
  - Email status
- Results display:
  - Leads found counter
  - Emails sent counter
  - Data table with all results
  - Success rate metric
- Export to CSV functionality
- Real-time logging display

## 2. Bulk Send (📦)
- Template selector with preview
- Email configuration:
  - From email selector
  - Reply-to field
  - Subject preview
- Send options:
  - All leads
  - Specific email
  - Leads from chosen search terms
  - Leads from search term groups
- Lead filtering:
  - Exclude previously contacted toggle
- Statistics display:
  - Total leads
  - Leads matching template language
  - Leads to be contacted
- Progress tracking:
  - Progress bar
  - Status updates
  - Results table
  - Success rate metric
- Email preview panel with HTML rendering

## 3. View Leads (👥)
- Dashboard metrics:
  - Total leads
  - Contacted leads
  - Conversion rate
- Interactive data table:
  - Search/filter functionality
  - Editable fields:
    - Email
    - First name
    - Last name
    - Company
    - Job title
  - Non-editable fields:
    - ID
    - Source
    - Last contact
    - Last email status
  - Delete checkbox
- Pagination controls
- Save changes button
- Export to CSV functionality
- Lead growth visualization
- Email campaign performance chart
- Detailed lead view expandable panels

## 4. Search Terms (🔑)
- Search terms dashboard metrics:
  - Total search terms
  - Total leads
  - Total emails sent
- Five main tabs:
  1. Search Term Groups:
    - Expandable group panels
    - Terms management with tags
    - Update button per group
  2. Performance:
    - Chart type selector (Bar/Pie)
    - Top 10 search terms visualization
    - Performance metrics table
  3. Add New Term:
    - Term input field
    - Group assignment dropdown
    - Add button
  4. AI Grouping:
    - Ungrouped terms counter
    - AI grouping trigger button
    - Progress indicator
  5. Manage Groups:
    - Create new group interface
    - Delete group interface
    - Group selector

## 5. Email Templates (✉️)
- Create new template:
  - Template name input
  - AI generation toggle
  - AI prompt input (if AI enabled)
  - Knowledge base integration toggle
  - Subject input
  - Body content editor
- Existing templates:
  - Expandable template panels
  - Edit fields:
    - Subject
    - AI customizable toggle
    - Body content
  - AI adjustment:
    - Prompt input
    - Apply button
  - Save changes button
  - Delete template button
  - Preview panel with HTML rendering
  - Language selector (ES/EN)

## 6. Projects & Campaigns (🚀)
- Projects section:
  - Add new project form
  - Project name input
- Campaigns section:
  - Add campaign form per project
  - Campaign name input
- Campaign settings:
  - Auto-send toggle
  - Loop automation toggle
  - AI customization toggle
  - Max emails per group input
  - Loop interval input
- Active project/campaign selector:
  - Project dropdown
  - Campaign dropdown
- Delete functionality for both
- Status indicators
- Validation messages

## 7. Knowledge Base (📚)
- Project selector
- Comprehensive form with sections:
  1. Basic Information:
    - KB name
    - KB bio
    - KB values
  2. Contact Information:
    - Contact name
    - Contact role
    - Contact email
  3. Company Information:
    - Company description
    - Company mission
    - Company target market
    - Company other details
  4. Product Information:
    - Product name
    - Product description
    - Product target customer
    - Product other details
  5. Additional Information:
    - Other context
    - Example email
- Save button
- Success/error messages
- Last updated indicator

## 8. AutoclientAI (🤖)
- Knowledge base information expander
- Additional context input
- Optimized search terms:
  - Generation button
  - Results display
- Automation controls:
  - Start/stop button
  - Status indicator
- Real-time monitoring:
  - Progress bar
  - Log container
  - Leads container
  - Analytics container
- Results display:
  - Total leads found
  - Total emails sent
  - Automation logs
  - Email sending logs
  - Success rate metrics

## 9. Automation Control (⚙️)
- Status dashboard:
  - Automation status metric
  - Start/stop button
- Quick scan functionality:
  - Scan button
  - Results display
- Real-time analytics:
  - Total leads metric
  - Emails sent metric
- Automation logs display
- Recently found leads display
- Progress indicators
- Error handling messages

## 10. Email Logs (📨)
- Email statistics:
  - Total emails sent
  - Success rate
- Date range selector:
  - Start date
  - End date
- Search functionality:
  - Email search
  - Subject search
- Metrics display:
  - Emails sent
  - Unique recipients
  - Success rate
- Daily email count chart
- Detailed log view:
  - Expandable entries
  - Full email preview
  - Status indicators
- Pagination controls
- Export to CSV functionality

## 11. Settings (🔄)
- General settings:
  - OpenAI API key
  - OpenAI API base URL
  - OpenAI model
- Email settings per account:
  - Name
  - Email address
  - Provider selection (SMTP/SES)
  - Provider-specific fields:
    SMTP:
      - Server
      - Port
      - Username
      - Password
    SES:
      - AWS access key
      - AWS secret key
      - AWS region
  - Test email functionality
  - Delete setting option
- Save buttons for each section
- Validation messages
- Success/error indicators

## 12. Sent Campaigns (📨)
- Campaigns data table:
  - ID
  - Sent timestamp
  - Recipient email
  - Template used
  - Subject
  - Content preview
  - Status
  - Message ID
  - Campaign ID
  - Lead information
  - Company information
- Content preview panel:
  - Full HTML rendering
  - Original content display
- Filtering capabilities
- Sorting options
- Export functionality

## Implementation Requirements
- Use Shadcn/UI components
- Maintain consistent styling
- Ensure mobile responsiveness
- Implement proper error handling
- Add loading states
- Include input validation
- Secure sensitive data
- Maintain current layout structure
- Preserve all database operations
- Include proper documentation