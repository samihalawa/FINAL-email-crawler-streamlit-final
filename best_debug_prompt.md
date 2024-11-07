FIRST PLEASE READ THE COMPLTE FILE AND ALL ITS LINES AND FUNCTIONS AND BLOCKS LINE AFTER LINE


[LINT OUTPUT]
@Lint errors  (important: do not limit yourself to linting and do everything here said. Use thi as reference for doing all the other ,steps. DO NOT EDIT LINITNG DIRECTLY)

[SYSTEM CONTEXT]
You are Claude. Analyze and fix ALL workflow breaks systematically. Output  all the steps of the follwing secuence in your reply and output for each ALL required fixes.

[ALWAYS ANALIZE THE 100% OF THE CODE LINE BY LINE]
Each reply and step and affirmation must be based solely on the existng current code here completely from start to end. ach of the lines of the code must be analyzed and considered for each step of the analysis. NEVR EVER PROPOSE FIXES THAT ARE REDUNDANT!!!


[ANALYSIS SEQUENCE]

 you must output each part titled and correctly outputed  and eacñ page and block muust be addressed



----> 1. MAP APPLICATION STRUCTURE
- Pages/Views
- Navigation flows  
- State dependencies
- Data requirements
- User permissions

Reply concisely to the question: Is an app like this one in this language adn framework usually have this structure and parts? Detect any issues and output edit Composer fix block if some very fatal  element  or part missing or structure is wrong (like for example but not limited to: not ending in main for Streamlit apps, or not ending in </html> for html apps, etc)

-----> 2. UNDEFINED FUNCTIONS AND CLASSES AND VARIABLES

Reply to the question: how many undefined classes and functions and linting issues are? only when 0 then u can reply to me.


----> 3 FULL LOGIC GRAPH FOR ALL THE RELATIONSHIPS AND ELEMENTS AND WORKFLOWS OF THE COMPLETE FILE SO TO DETECT missing things adn broken things and missing and unreachable things etc

Fix: [Composer formatted edit fix for all the problmes derived from the Full Logic Graph ccording to how you have been told before Composer format works.]


---->4. LIST ALL HALLUCINATIONS AND NOT EXISIITNG MISNAME PACKAGES AND ATTRIBUTES AND LIBRARIES 

Fix: [Composer formatted edit fix for all from the list according to how you have been told before Composer format works.]

IF PLACEHOLDERS ALREADY IN THE CODE REPLACE ALL MISSING FUNCTIONALITY OR PLACEHOLDERS MISSING WITH FINAL LOGIC 

----> 5.  LIST ALL PLACEHOLDRS AND CODE LINES 
OUTPUT COMPOSR EDIT BLOCK  REPLACE THEM WITH FINAL COMPACT LOGIC THAT WORKS WITH THE REST OF THE FILE 

----> 6. FOR EACH PAGE/VIEW:

PREREQUISITES
Auth: [required state]
Data: [required data]
State: [required state]
Access: [required permissions]

UI ELEMENTS 
Buttons: [list]
Inputs: [list]
Display: [list]
Navigation: [list]

WORKFLOW CHAINS OUTPUT IT COMPLET AND WITHOUT ASSUMPTIONS FOR EACH AND EVERY VIEW 
Element -> Action -> Handler -> State -> UI
Input -> Validate -> Process -> Save -> Show
Submit -> Verify -> Update -> Notify -> Next

VALIDATION
For each chain:
✓ Elements present
✓ Handlers exist
✓ State managed
✓ Feedback shown
✗ Missing/broken

3. IDENTIFY ALL BREAKS
For each ✗:
Break: [what's missing]
At: [file:line]
Chain: [broken workflow]
Impact: [what fails]
Proof: [why it breaks]
Fix: [Composer formatted edit fix according to how you have been told before Composer format works.]
Note: The Composer format edit block must adjust all parts of the code related to what we are fixing here. Always include exact line number and action (add / remove / update - fix) in the comment at the top of each edit block. 

4. OUTPUT FORMAT FOR EACH PAGE

PAGE ANALYSIS: [name]

Current Structure:
[Visual flow diagram]

Required Chains:
[Element/action mapping]

Missing Components:
1. Break: [description]
   Location: [file:line]
   Impact: [workflow break]
   Fix: [code block]

2. Break: [next issue]
   Location: [file:line] 
   Impact: [workflow break]
   Fix: [code block]

Continue until ALL breaks fixed.

[VALIDATION RULES]
Must verify:
1. Complete UI flows
2. All handlers exist
3. All state managed
4. All validation present
5. All feedback shown
6. All errors handled
7. All navigation works
8. All data validated
9. All security enforced
10. All resources managed

[RESPONSE RULES]
1. Analyze ALL pages
2. Find ALL breaks
3. Fix ALL issues
4. Show ALL chains
5. Verify ALL fixes
6. No enhancements
7. Only fix proven breaks
8. Clear impacts shown
9. Minimal fixes only
10. Complete validation

[STOP CONDITIONS]
- All pages analyzed
- All flows validated  
- All breaks fixed
- All chains complete
- All states verified

[RESPONSE START]
"Beginning complete workflow analysis.  First , i will include a list numbered with all the pages and componenents numbered that i will be analyzing and processing. After it i will process each of them one after one until the end ."

[COMPLETION]
"All pages and workflows validated and fixed. Each Line of th code has been analyzed and there are not any other problms and th ecode is 100% bugless and problem free . Bye! "

[PROMPTING TRICKS]
/force_complete_analysis = true
/require_all_fixes = true
/prevent_partial_response = true
/ensure_minimal_fixes = true
/validate_full_chains = true
/block_enhancements = true
/focus_on_breaks = true
/clear_impacts_only = true
/sequential_fixes = true
/verify_completion = true

Important: output only Composr edit codblocks with other code... comments in stead of complete blcoks. You must meet the Composer format valid criteria, using one line comments for defining instructions etc. You know.

Read the comeplte file first, list all the pages and views one after the other no mater if need adjsutment or not. and then you start one after the other until the end. Explicitely hihjlight Number out of total (3 / 10) and repeat at the end of each page procsed, befor ethe titile fo the next one "Page processed! Now only X pages remaining. Let's process the next one!"

DONTS:
DONT ASK THINGS LIKE "Would you like me to continue .."
BUT CONTINUE ONE AFTER THE OTHER WIHTOUT FOCUSING ON LIMITS OR ANY OTHER THINGS. YOU CANNOT ASK QUESTIONS AT ALL 
DONT USE PLACEHOLDERS LIKE [Continuing with remaining pages...] !! BUT OUTPUT ALL 

YOUR MESSAGE MUST ALWAYS END WITH "All pages and workflows validated and fixed. Each Line of th code has been analyzed and there are not any other problms and th ecode is 100% bugless and problem free . Bye! "

YOU CAN ONLY OUTPUT ONE MESSAGE 














[SYSTEM CONTEXT]
You are Claude. Analyze and fix ALL workflow breaks systematically. Output  all the steps of the follwing secuence in your reply and output for each ALL required fixes.

[ANALYSIS SEQUENCE] you must output each part titled and correctly outputed 

🤖🤖🤖 ----> 1. MAP APPLICATION STRUCTURE
- Pages/Views
- Navigation flows  
- State dependencies
- Data requirements
- User permissions


🤖🤖🤖----> 2 .FULL LOGIC GRAPH FOR ALL THE RELATIONSHIPS AND ELEMENTS AND WORKFLOWS OF THE COMPLETE FILE SO TO DETECT missing things adn broken things and missing and unreachable things etc

Fix: [Composer formatted edit fix for all the problmes derived from the Full Logic Graph ccording to how you have been told before Composer format works.]


🤖🤖🤖 ---->3. LIST ALL HALLUCINATIONS AND NOT EXISIITNG MISNAME PACKAGES AND ATTRIBUTES AND LIBRARIES 

Fix: [Composer formatted edit fix for all from the list according to how you have been told before Composer format works.]


🤖🤖🤖----->4. LAST CHECK OF ALL THE SCRIPT


ANALIZE THE COMPLETE CODE LINE BY LINE FIRST AND REPLACE ALL MISSING LOGIC AND PLACEHOLDERS FIRST and any other issue if any


🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 SUPER IMPORTANT: READ THE COMPLETE FILE FROM FIRST TO LAST LINE , READING LINE AFTER LINE AS YOU USUALLY MAKE THE MISTAKE OF TELLING ME TO FIX THINGS THAT ARE NOT MISING/FIX NEEDING, just because you did not read the complete file. Read and make sure each and every fix you ourput is 100% necessary and is not in the currenct code in any part of it.
This is important for each of the times you search or anaylize the code:
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 


🤖🤖🤖 ----> 5 . FOR EACH PAGE/VIEW:

//////FOR EACH PAGE - SECTION START

PREREQUISITES
Auth: [required state]
Data: [required data]
State: [required state]
Access: [required permissions]

UI ELEMENTS 
Buttons: [list]
Inputs: [list]
Display: [list]
Navigation: [list]

WORKFLOW CHAINS OUTPUT IT COMPLET AND WITHOUT ASSUMPTIONS FOR EACH AND EVERY VIEW 
Element -> Action -> Handler -> State -> UI
Input -> Validate -> Process -> Save -> Show
Submit -> Verify -> Update -> Notify -> Next

VALIDATION
For each chain:
✓ Elements present
✓ Handlers exist
✓ State managed
✓ Feedback shown
✗ Missing/broken

3. IDENTIFY ALL BREAKS
For each ✗:
Break: [what's missing]
At: [file:line]
Chain: [broken workflow]
Impact: [what fails]
Proof: [why it breaks]
Fix: [Composer formatted edit fix according to how you have been told before Composer format works.]
Note: The Composer format edit block must adjust all parts of the code related to what we are fixing here. Always include exact line number and action (add / remove / update - fix) in the comment at the top of each edit block. 

4. OUTPUT FORMAT FOR EACH PAGE

PAGE ANALYSIS: [name]

Current Structure:
[Visual flow diagram]

Required Chains:
[Element/action mapping]

Missing Components:
1. Break: [description]
   Location: [file:line]
   Impact: [workflow break]
   Fix: [code block]

2. Break: [next issue]
   Location: [file:line] 
   Impact: [workflow break]
   Fix: [code block]

Continue until ALL breaks fixed.

//////FOR EACH PAGE - SECTION END



[VALIDATION RULES]
Must verify:
1. Complete UI flows
2. All handlers exist
3. All state managed
4. All validation present
5. All feedback shown
6. All errors handled
7. All navigation works
8. All data validated
9. All security enforced
10. All resources managed

[RESPONSE RULES]
1. Analyze ALL pages
2. Find ALL breaks
3. Fix ALL issues
4. Show ALL chains
5. Verify ALL fixes
6. No enhancements
7. Only fix proven breaks
8. Clear impacts shown
9. Minimal fixes only
10. Complete validation

[STOP CONDITIONS]
- All pages analyzed
- All flows validated  
- All breaks fixed
- All chains complete
- All states verified

[RESPONSE START]
"Beginning complete workflow analysis.  First , i will include a list numbered with all the pages and componenents numbered that i will be analyzing and processing. After it i will process each of them one after one until the end ."

[COMPLETION]
"All pages and workflows validated and fixed. Each Line of th code has been analyzed and there are not any other problms and th ecode is 100% bugless and problem free . Bye! "

[PROMPTING TRICKS]
/force_complete_analysis = true
/require_all_fixes = true
/prevent_partial_response = true
/ensure_minimal_fixes = true
/validate_full_chains = true
/block_enhancements = true
/focus_on_breaks = true
/clear_impacts_only = true
/sequential_fixes = true
/verify_completion = true

Important: output only Composr edit codblocks with other code... comments in stead of complete blcoks. You must meet the Composer format valid criteria, using one line comments for defining instructions etc. You know.

Read the comeplte file first, list all the pages and views one after the other no mater if need adjsutment or not. and then you start one after the other until the end. Explicitely hihjlight Number out of total (3 / 10) and repeat at the end of each page procsed, befor ethe titile fo the next one "Page processed! Now only X pages remaining. Let's process the next one!"



[DONTS]

DONT hallucinating : your reply must be 100% accurate and true (for example dont suggest edits that are not needed-already implemented in original code)


DONT ASK THINGS LIKE "Would you like me to continue .."
BUT CONTINUE ONE AFTER THE OTHER WIHTOUT FOCUSING ON LIMITS OR ANY OTHER THINGS. YOU CANNOT ASK QUESTIONS AT ALL 
DONT USE PLACEHOLDERS LIKE [Continuing with remaining pages...] !! BUT OUTPUT ALL 

YOUR MESSAGE MUST ALWAYS END WITH THIS CONFIRMATION WHCIH MUST BE TRUE 
"All pages and workflows validated and fixed. Each Line of th code has been analyzed and there are not any other problms and th ecode is 100% bugless and problem free . 
I confirm that all the edits said in thsi message are fact checked and 100% necessary for the current codebase.
 Bye! "






YOU CAN ONLY OUTPUT ONE MESSAGE AND MUST BE ACCURATE, FACT CHECKED AND ADDRSSING EACH AND EVERYTHING TO THE BEST WITHOUT PLACEHOLDERS NOR ASSUMPTIONS



READ LINE AFTER LINE BEFORE REPLY, FROM FIRST TO LAST LINE OF THE FILE