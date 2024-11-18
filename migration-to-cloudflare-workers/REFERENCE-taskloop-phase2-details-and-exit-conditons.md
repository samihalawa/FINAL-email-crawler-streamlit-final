
## Automation Process Logic
# Phase 2: Automation Process Distribution Requirements

## Core Infrastructure
1. Components Distribution:
   - Python email search process: HuggingFace Space (https://luigi12345-python-worker.hf.space)
   - Long-running operations: HuggingFace Space
   - Frontend: Cloudflare Pages 
   - Coordination layer: Cloudflare Worker (https://autoclient-worker.trigox.workers.dev)
   - Database: 

## Database Configuration
'''sql
CREATE TABLE IF NOT EXISTS automation_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status VARCHAR(50) NOT NULL,
    current_group VARCHAR(255),
    current_position INTEGER,
    total_emails_sent INTEGER DEFAULT 0,
    group_emails_sent INTEGER DEFAULT 0,
    distribution_method VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    loop_interval INTEGER,
    max_emails_per_group INTEGER,
    loop_automation BOOLEAN DEFAULT false
);
'''


### Required Tables
1. Process State Table:

### Search Term Processing
1. Batch Operation:
   - Process search terms in groups
   - Generate variations within groups
   - Execute manual search process
   - Handle lead finding and emailing

### Control Parameters
1. loop_interval
   - Definition: Maximum leads per search term
   - Example: At 40, moves to next term after finding 40 leads
   - Purpose: Controls iteration timing

2. max_emails_per_group
   - Definition: Email limit per group per run
   - Example: At 300, stops group after 300 emails
   - Purpose: Resource distribution control
   - Behavior: Moves to next group when limit reached

3. Distribution Methods
   - Random: Randomly selects next group
   - Equitable: Fairly distributes across groups
   - Configurable from Automation AI view

4. loop_automation
   - Definition: Enables continuous processing
   - Behavior: Restarts after all groups completed
   - Configurable from Automation AI view

### Process Control
1. Required Capabilities:
   - Start/Stop at any time
   - Resume from stopped point
   - Maintain process state via Supabase
   - State synchronization between Worker and HuggingFace Space

### Visual Process Display
1. Core Requirements:
   - Dynamic log display
   - Real-time process updates via Worker
   - Shows current operation status
   - Displays progress information

2. Display Characteristics:
   - Updates in real-time
   - Shows process flow
   - Maintains visual continuity
   - Provides clear status indication

## Success Criteria

### Functional Requirements
1. Batch Processing:
   - Respects loop_interval exactly
   - Enforces max_emails_per_group
   - Implements both distribution methods
   - Maintains group processing order

2. State Management:
   - Preserves current position in Supabase
   - Enables exact resumption
   - Tracks group progress
   - Maintains email counts

3. Process Display:
   - Shows real-time updates
   - Displays current operation
   - Indicates group progress
   - Presents clear status



### Required-Working
1. Distribution:
   - HuggingFace Space properly deployed and accessible
   - Cloudflare Worker communication established
   - Supabase state synchronization working

2. Process Control:
   - Start/Stop functioning through Worker API
   - Resume working correctly with state from Supabase
   - State preserved accurately in database

3. Parameter Management:
   - loop_interval enforced
   - max_emails_per_group respected
   - Distribution methods working
   - loop_automation functioning

### Verification Steps
1. Process Testing:
   - Verify batch processing
   - Test state preservation in Supabase
   - Validate resumption
   - Check distribution methods

2. Display Testing:
   - Verify real-time updates
   - Check progress indication
   - Test status display
   - Validate process visualization



## Exit Conditions BEFORE WHICH YOU CANNOT ASK QUESTION
if conditions!==Required-Working: 
    human-in-the-loop=false
    keep-adjusting-code=true