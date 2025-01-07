def main():
    initialize_session_state()
    
    try:
        st.set_page_config(
            page_title="Autoclient.ai | Lead Generation AI App",
            layout="wide",
            initial_sidebar_state="expanded",
            page_icon=""
        )

        st.sidebar.title("AutoclientAI")
        st.sidebar.markdown("Select a page to navigate through the application.")

        pages = {
            "🔍 Manual Search": manual_search_page,
            "📦 Bulk Send": bulk_send_page,
            "👥 View Leads": view_leads_page,
            "🔑 Search Terms": search_terms_page,
            "✉️ Email Templates": email_templates_page,
            "🚀 Projects & Campaigns": projects_campaigns_page,
            "📚 Knowledge Base": knowledge_base_page,
            "🤖 AutoclientAI": autoclient_ai_page,
            "⚙️ Automation Control": automation_control_panel_page,
            "📨 Email Logs": view_campaign_logs,
            "🔄 Settings": settings_page,
            "📨 Sent Campaigns": view_sent_email_campaigns
        }

        if "current_page" not in st.session_state:
            st.session_state.current_page = "🔍 Manual Search"

        # Define callback before creating the widget
        def on_page_change():
            st.session_state.current_page = st.session_state.selected_option

        with st.sidebar:
            option_menu(
                menu_title="Navigation",
                options=list(pages.keys()),
                icons=["search", "send", "people", "key", "envelope", "folder", "book", "robot", "gear", "list-check", "gear", "envelope-open"],
                menu_icon="cast",
                default_index=list(pages.keys()).index(st.session_state.current_page),
                on_change=on_page_change,
                key='selected_option'
            )

        # Initialize background process on first load
        if 'background_process_started' not in st.session_state:
            start_background_process()
            st.session_state.background_process_started = True

        # Run the selected page
        pages[st.session_state.current_page]()

        st.sidebar.markdown("---")
        st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.exception("An error occurred in main()")
        
    finally:
        # Ensure cleanup happens when the app is closed
        atexit.register(cleanup_resources)

def cleanup_resources():
    """Cleanup resources when the app is closed"""
    if 'initialized' in st.session_state and st.session_state.initialized:
        stop_background_process()
        cleanup_temp_files()

def cleanup_temp_files():
    """Clean up any temporary files created during the session"""
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        try:
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp directory: {e}")

if __name__ == "__main__":
    main()
