# Detailed AutoclientAI Phase 2 Enhancement Requirements

## Original System Understanding
1. Current Implementation:
   - Manual search page with email capability
   - Search term groups managed in Search Terms view
   - Existing Supabase database with Campaign model
   - React frontend ported from Streamlit

2. Database Schema Context:
   ```python
   class Campaign(Base):
       loop_automation = Column(Boolean, default=False)
       max_emails_per_group = Column(BigInteger, default=40)
       loop_interval = Column(BigInteger, default=60)
   ```

## Distribution Requirements
1. Component Migration:
   - Move Python search/email process to HuggingFace Space
   - Keep frontend on Cloudflare Pages
   - Deploy coordinator to Cloudflare Workers
   - Migrate only automation-related processes

2. What Stays Manual:
   - Bulk send page functionality
   - Manual search page functionality
   - Search term group creation/management
   - All existing UI outside AutoclientAI view

## Core Process Logic

### Search Term Processing
1. Group Processing Rules:
   - Process search terms within their groups
   - Use manual_search_page logic with email=ON
   - Count only when:
     - Lead is found AND
     - Email is successfully sent
   - Move to next term when loop_interval reached

2. Exact Counting Logic:
   - loop_interval (e.g., 40):
     - Counts successful lead finds + emails
     - Moves to next search term after count reached
   - max_emails_per_group (e.g., 300):
     - Tracks emails per entire group
     - Moves to next group when limit reached
     - Unprocessed terms get priority next loop

3. Distribution Methods:
   - Random:
     - Simply selects next group randomly
   - Equitable:
     - Calculates: max_emails_per_group ÷ terms_in_group
     - Attempts equal distribution per term
     - Adjusts if terms exhaust leads

4. AI Search Term Enhancement:
   - Prompt: "You are an AI marketing automation tool expert..."
   - Input: Group name + existing terms
   - Purpose: Generate optimized variations
   - Context: Consider group objectives
   - Integration: Process variations in same batch

## AutoclientAI View Specifics

### Visual Requirements
1. Process Display:
   - Exactly like manual_search_page logs but automated
   - Collapsible search term groups
   - Currently processing term highlighted
   - Auto-scrolling detailed logs showing:
     - Current search URL
     - Found lead details
     - Email sending status
     - Batch progress

2. Animation Requirements:
   - Smooth group transitions
   - Term highlighting/unhighlighting
   - Continuous log updates
   - Progress indicators
   - Visual process flow

### Control Panel
1. Required Controls:
   - Start/Stop automation
   - Pause/Resume processing
   - loop_interval input
   - max_emails_per_group input
   - distribution_method selector
   - loop_automation toggle

2. Control Behavior:
   - Start: Begins from current position
   - Stop: Complete halt, preserves position
   - Pause: Temporary hold, maintains state
   - Resume: Continues from exact position

## State Management

### Required State Tracking
1. Per Group:
   - Emails sent count
   - Terms processed
   - Current position
   - Processing status

2. Per Term:
   - Leads found
   - Emails sent
   - Processing status
   - Priority flag

### State Preservation
1. Must Preserve:
   - Current search term position
   - Current group position
   - All counters
   - Distribution state
   - Process status

2. Resume Capability:
   - Exact position recovery
   - Counter preservation
   - State restoration
   - Progress continuation

## Process Flow Details

### Batch Processing
1. Normal Flow:
   - Process terms until loop_interval
   - Track group emails until max_emails_per_group
   - Move to next group based on distribution_method
   - Continue until all groups processed

2. loop_automation Behavior:
   - If enabled: restart after completion
   - If disabled: stop after one cycle
   - Preserve unprocessed terms priority

### Error Handling
1. Must Handle:
   - Search failures
   - Email failures
   - State loss
   - Connection issues
   - Process interruptions

## Success Criteria

### Functional Requirements
1. Process Verification:
   - Follows manual_search_page logic exactly
   - Maintains all counts accurately
   - Preserves group order
   - Handles priorities correctly

2. Visual Verification:
   - Real-time log updates
   - Smooth animations
   - Clear status indication
   - Accurate progress display

3. State Verification:
   - Perfect state preservation
   - Accurate resumption
   - Counter accuracy
   - Priority maintenance

### Exit Conditions
1. Process Testing:
   - Complete cycle verification
   - All counts validated
   - All states preserved
   - All transitions smooth

2. Visual Testing:
   - Log display accuracy
   - Animation smoothness
   - Status clarity
   - Progress indication

3. State Testing:
   - Position preservation
   - Counter accuracy
   - Priority handling
   - Resume functionality
