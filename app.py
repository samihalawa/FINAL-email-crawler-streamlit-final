from datetime import datetime
import streamlit as st

def update_log(log_container, message, level='info'):
    icon = {
        'info': '🔵',
        'success': '🟢',
        'warning': '🟠',
        'error': '🔴',
        'email_sent': '🟣'
    }.get(level, '⚪')
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"{icon} [{timestamp}] {message}"
    
    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []
    
    st.session_state.log_entries.append(log_entry)
    
    # Keep only last 1000 entries to prevent memory issues
    if len(st.session_state.log_entries) > 1000:
        st.session_state.log_entries = st.session_state.log_entries[-1000:]
    
    # Create scrollable log container with auto-scroll and improved styling
    log_html = f"""
        <style>
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .log-container {{
                background: rgba(0,0,0,0.05);
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            
            .log-scroll {{
                height: 400px;
                overflow-y: auto;
                font-family: 'Monaco', 'Consolas', monospace;
                font-size: 0.9em;
                line-height: 1.5;
                padding: 15px;
                background: rgba(255,255,255,0.9);
                border-radius: 8px;
                scroll-behavior: smooth;
            }}
            
            .log-entry {{
                padding: 8px 12px;
                margin-bottom: 8px;
                border-radius: 6px;
                background: rgba(0,0,0,0.02);
                border-left: 3px solid #1E88E5;
                animation: fadeIn 0.3s ease-out forwards;
            }}
            
            .log-entry:hover {{
                background: rgba(0,0,0,0.04);
                transform: translateX(5px);
                transition: all 0.2s ease;
            }}
        </style>
        
        <div class="log-container">
            <div id="log-scroll" class="log-scroll">
                {"".join(f'<div class="log-entry">{entry}</div>' for entry in st.session_state.log_entries)}
            </div>
        </div>
        
        <script>
            function scrollToBottom() {{
                const container = document.getElementById('log-scroll');
                if (container) {{
                    container.scrollTop = container.scrollHeight;
                }}
            }}
            
            // Initial scroll
            scrollToBottom();
            
            // Set up a mutation observer to watch for changes
            const observer = new MutationObserver(scrollToBottom);
            const container = document.getElementById('log-scroll');
            if (container) {{
                observer.observe(container, {{ 
                    childList: true, 
                    subtree: true,
                    characterData: true
                }});
            }}
            
            // Additional smooth scroll on new entries
            document.querySelectorAll('.log-entry').forEach(entry => {{
                entry.addEventListener('animationend', scrollToBottom);
            }});
        </script>
    """
    log_container.markdown(log_html, unsafe_allow_html=True) 