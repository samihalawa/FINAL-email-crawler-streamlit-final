# Phase 2: Automation Process Distribution Requirements

## Core Distribution
1. Components to Move:
   - Python email search process to HuggingFace Space
   - Long-running operations to HuggingFace Space
   - Frontend remains on Cloudflare Pages
   - Coordination layer on Cloudflare Workers

## Automation Process Logic

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
   - Maintain process state

### Visual Process Display
1. Core Requirements:
   - Dynamic log display
   - Real-time process updates
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
   - Preserves current position
   - Enables exact resumption
   - Tracks group progress
   - Maintains email counts

3. Process Display:
   - Shows real-time updates
   - Displays current operation
   - Indicates group progress
   - Presents clear status

## Exit Conditions

### Required Working
1. Distribution:
   - All components properly deployed
   - Communication established
   - State synchronization working

2. Process Control:
   - Start/Stop functioning
   - Resume working correctly
   - State preserved accurately

3. Parameter Management:
   - loop_interval enforced
   - max_emails_per_group respected
   - Distribution methods working
   - loop_automation functioning

### Verification Steps
1. Process Testing:
   - Verify batch processing
   - Test state preservation
   - Validate resumption
   - Check distribution methods

2. Display Testing:
   - Verify real-time updates
   - Check progress indication
   - Test status display
   - Validate process visualization 