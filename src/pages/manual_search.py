from nicegui import ui
from typing import List, Optional
from core.database import get_db
from core.auth import auth_handler
from core.background import task_manager
from services.search import SearchService
from services.email import EmailService

search_service = SearchService()
email_service = EmailService()

@ui.page('/manual-search')
@auth_handler.require_auth
async def page():
    # State management
    search_terms: List[str] = []
    results_container = None
    progress = None
    
    async def add_search_term(term: str):
        if term and term not in search_terms:
            search_terms.append(term)
            update_search_terms_display()
    
    def update_search_terms_display():
        with terms_container:
            terms_container.clear()
            for term in search_terms:
                with ui.row().classes('items-center w-full'):
                    ui.label(term)
                    ui.button(icon='close', on_click=lambda t=term: remove_term(t)).props('flat round')
    
    def remove_term(term: str):
        search_terms.remove(term)
        update_search_terms_display()
    
    async def start_search():
        if not search_terms:
            ui.notify('Please add at least one search term', type='warning')
            return
            
        # Create progress indicators
        nonlocal progress
        progress = ui.progress()
        
        # Start background task
        task_id = await task_manager.add_task(
            search_service.perform_search,
            search_terms=search_terms,
            num_results=num_results.value,
            language=language.value,
            optimize_english=optimize_english.value,
            optimize_spanish=optimize_spanish.value,
            shuffle_keywords=shuffle_keywords.value
        )
        
        # Monitor task progress
        while True:
            task = task_manager.get_task(task_id)
            if task.status == "completed":
                display_results(task.result)
                break
            elif task.status == "failed":
                ui.notify(f"Search failed: {task.error}", type='negative')
                break
            await ui.sleep(1)
    
    def display_results(results):
        nonlocal results_container
        if results_container:
            results_container.clear()
        
        with results_container:
            ui.label(f'Found {len(results)} leads').classes('text-h6')
            
            with ui.table().classes('w-full').props('rows-per-page-options=[10,20,50]'):
                ui.table_column('Email', 'email')
                ui.table_column('Name', 'name')
                ui.table_column('Company', 'company')
                ui.table_column('Source', 'source_url')
                ui.table_column('Actions', 'actions')
                
                rows = []
                for result in results:
                    row = {
                        'email': result['email'],
                        'name': result.get('name', ''),
                        'company': result.get('company', ''),
                        'source_url': result['url'],
                        'actions': ui.button('Send Email', on_click=lambda r=result: send_email(r))
                    }
                    rows.append(row)
                
                ui.table_rows(rows)
    
    async def send_email(lead):
        # Implement email sending logic
        pass
    
    # UI Layout
    ui.label('Manual Search').classes('text-h4 q-ma-md')
    
    with ui.card().classes('w-full'):
        with ui.row().classes('w-full items-center'):
            term_input = ui.input('Enter search term').props('outlined')
            ui.button('Add', on_click=lambda: add_search_term(term_input.value))
        
        terms_container = ui.column().classes('w-full q-pa-md')
        
        with ui.row().classes('w-full q-pa-md'):
            with ui.column().classes('w-2/3'):
                num_results = ui.number('Results per term', value=10, min=1, max=100)
            with ui.column().classes('w-1/3'):
                optimize_english = ui.checkbox('Optimize (English)')
                optimize_spanish = ui.checkbox('Optimize (Spanish)')
                shuffle_keywords = ui.checkbox('Shuffle Keywords')
                language = ui.select(
                    'Language',
                    options=[
                        {'label': 'English', 'value': 'EN'},
                        {'label': 'Spanish', 'value': 'ES'}
                    ],
                    value='EN'
                )
        
        ui.button('Start Search', on_click=start_search).props('primary').classes('q-ma-md')
    
    results_container = ui.card().classes('w-full q-ma-md') 