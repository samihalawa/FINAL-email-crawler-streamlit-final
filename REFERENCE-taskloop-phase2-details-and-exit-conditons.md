# Phase 2: Automation Process Distribution Requirements

## Core Process Understanding

### Manual Operations Remaining Manual
- Bulk send page functionality
- Manual search page functionality
- Search term group creation/management (in Search Terms view)

### AI Search Term Enhancement
- Input: Search term group name + existing terms
- Role: Marketing automation optimization
- Purpose: Generate related, optimized variations
- Context: Consider group name and existing terms
- Output: Optimized terms for batch processing

### Process Flow Specifics
1. Search & Email Process
   - Uses manual_search_page logic with email=ON
   - Counts as "processed" when:
     - Lead found AND
     - Email successfully sent
   - Moves to next term at loop_interval count

2. Group Processing
   - Process search terms within group until:
     - max_emails_per_group reached OR
     - all terms processed
   - Unprocessed terms get priority next loop
   - Terms processed in original group order

3. Distribution Methods
   - Random: Simple next group selection
   - Equitable: 
     - Calculate: max_emails_per_group ÷ terms_in_group
     - Try reaching this count for each term
     - Adjust if some terms exhaust leads

### AutoclientAI View Requirements
1. Visual Process Display (like manual_search_page but automated)
   - Collapsible search term groups
   - Currently processing term highlighted
   - Auto-scrolling detailed logs showing:
     - Current search URL
     - Found lead details
     - Email sending status
     - Just like manual_search_page logs

2. Real-time Updates
   - Smooth group transitions
   - Term highlighting/unhighlighting
   - Continuous log updates
   - Group progress indicators

3. Controls
   - Start/Stop automation
   - Pause/Resume processing
   - loop_interval setting
   - max_emails_per_group setting
   - distribution_method selection
   - loop_automation toggle

## Technical Requirements

### Component Distribution
1. HuggingFace Space
   - Python search/email process
   - Uses manual_search logic
   - Maintains existing email sending

2. Cloudflare Workers
   - State coordination
   - Process management
   - Group transitions

3. Cloudflare Pages
   - Existing React UI
   - Enhanced AutoclientAI view
   - Real-time visualization

### State Management
1. Track Per Group
   - Emails sent count
   - Terms processed
   - Current position

2. Track Per Term
   - Leads found
   - Emails sent
   - Processing status

## Exit Conditions

### Must Verify
1. Process Behavior
   - Follows manual_search_page logic
   - Respects all limits
   - Maintains group order
   - Prioritizes correctly

2. Visual Display
   - Shows real-time logs
   - Highlights active terms
   - Updates smoothly
   - Matches manual_search_page style

3. State Management
   - Maintains counts accurately
   - Preserves position on pause
   - Resumes correctly
   - Tracks all progress

### Final Verification
1. Process complete test cycle
2. Verify all counts match
3. Confirm visual updates
4. Test pause/resume
5. Validate distribution 