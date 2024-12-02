from nicegui import ui, app
from pathlib import Path
import asyncio
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

# Internal imports
from core.config import settings
from core.database import init_db, get_db
from core.auth import auth_middleware
from core.background import BackgroundTaskManager
from pages import (
    manual_search,
    bulk_send,
    view_leads,
    search_terms,
    email_templates,
    projects,
    knowledge_base,
    autoclient_ai,
    automation_control,
    email_logs,
    settings_page
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize background task manager
task_manager = BackgroundTaskManager()

@asynccontextmanager
async def lifespan(app):
    # Startup
    await init_db()
    await task_manager.start()
    yield
    # Shutdown
    await task_manager.stop()

# Setup authentication
auth_handler = auth_middleware.AuthHandler()

# Main navigation setup
def setup_navigation():
    with ui.header().classes('items-center justify-between'):
        ui.button(on_click=lambda: left_drawer.toggle()).props('flat color=white icon=menu')
        ui.label('AutoClient.ai').classes('text-h6')
        
    with ui.left_drawer().classes('bg-blue-100') as left_drawer:
        ui.label('Navigation').classes('text-h6 q-pa-md')
        with ui.list():
            ui.link('🔍 Manual Search', manual_search.page)
            ui.link('📦 Bulk Send', bulk_send.page)
            ui.link('👥 View Leads', view_leads.page)
            ui.link('🔑 Search Terms', search_terms.page)
            ui.link('✉️ Email Templates', email_templates.page)
            ui.link('🚀 Projects & Campaigns', projects.page)
            ui.link('📚 Knowledge Base', knowledge_base.page)
            ui.link('🤖 AutoclientAI', autoclient_ai.page)
            ui.link('⚙️ Automation Control', automation_control.page)
            ui.link('📨 Email Logs', email_logs.page)
            ui.link('🔄 Settings', settings_page.page)

# Main application setup
def init_app():
    app.include_router(auth_middleware.router)
    
    @ui.page('/')
    @auth_handler.require_auth
    def main_page():
        setup_navigation()
        with ui.column().classes('w-full items-center'):
            ui.label('Welcome to AutoClient.ai').classes('text-h4 q-ma-md')
            ui.label('Select an option from the menu to get started').classes('text-subtitle1 q-ma-sm')

    ui.run(
        title='AutoClient.ai',
        favicon='🤖',
        dark=True,
        reload=False,
        show=False,
        port=settings.PORT,
        host=settings.HOST
    )

if __name__ == "__main__":
    init_app() 