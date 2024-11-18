Convert an existing Streamlit application into a distributed web application while maintaining identical functionality and data models. The system should preserve all 12 core pages including manual search, bulk operations, lead management, and automation controls.

Core Features:
- Exact feature parity with original Streamlit app across all 12 pages
- Complete lead management and email automation system
- Knowledge base and template management
- Project and campaign tracking capabilities
- Search term optimization and bulk operations
- Email logging and campaign analytics
- Settings and automation control panels

Technical Requirements:
- Pre-built UI components only
- Background task queue system
- AI parameter optimization
- Automated scheduling
- Preserve all existing database models and relationships and Supabase integration
- Integration with googlesearch-python package

UI/Style using CDN packages and never from scratch if somethign there is an importable theme of componenets prebuilt that in a few lines cna be used. Must look modern, not boostrap not Tailwind pure: components and blocks preexisting! 
- Workflow-optimized layout with contextual navigation
- Professional color scheme with subtle accent colors for different functional areas  
- Clear visual feedback for automation states and system processes. USE CDN resorcues for every piece of funcitonality int he front end that can be importable so not even the notifications or toasts or anythgin must be never from scratch


## Final Notes

The migration should result in a seamlessly functioning **Autoclient.ai** application that mirrors the exact user experience of the original Streamlit app. By utilizing optimized, pre-built components without compromising on existing functionalities or user experience.

Ensure that every aspect of the original streamlit_run.py application is thoroughly analyzed and replicated in the new architecture, leaving no room for missing features or broken functionalities. 