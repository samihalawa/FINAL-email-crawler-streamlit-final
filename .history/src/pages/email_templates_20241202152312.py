from nicegui import ui
from typing import Dict, Any, Optional
from datetime import datetime

from core.database import get_db, EmailTemplate
from core.auth import auth_handler
from services.ai import AIService

ai_service = AIService()

@ui.page('/email-templates')
@auth_handler.require_auth
async def page():
    current_template: Optional[EmailTemplate] = None
    
    async def load_templates():
        async with get_db() as session:
            templates = await session.execute(
                "SELECT * FROM email_templates ORDER BY created_at DESC"
            )
            return [dict(t) for t in templates]
    
    async def save_template():
        async with get_db() as session:
            if current_template and current_template.id:
                template = await session.get(EmailTemplate, current_template.id)
                template.name = name_input.value
                template.subject = subject_input.value
                template.body_content = body_editor.value
                template.is_ai_customizable = ai_checkbox.value
            else:
                template = EmailTemplate(
                    name=name_input.value,
                    subject=subject_input.value,
                    body_content=body_editor.value,
                    is_ai_customizable=ai_checkbox.value,
                    campaign_id=campaign_select.value
                )
                session.add(template)
            await session.commit()
            ui.notify('Template saved successfully')
            await refresh_template_list()
    
    async def delete_template(template_id: int):
        async with get_db() as session:
            template = await session.get(EmailTemplate, template_id)
            if template:
                await session.delete(template)
                await session.commit()
                ui.notify('Template deleted successfully')
                await refresh_template_list()
    
    async def optimize_template():
        if not current_template:
            ui.notify('Please select a template first', type='warning')
            return
            
        with ui.loading('Optimizing template...'):
            # Get knowledge base for context
            async with get_db() as session:
                kb = await session.get_knowledge_base()
                if not kb:
                    ui.notify('Knowledge base not found', type='negative')
                    return
                
                # Generate optimized content
                optimized = await ai_service.optimize_email_template(
                    current_template,
                    None,  # No specific lead for general optimization
                    kb
                )
                
                # Show optimization dialog
                with optimization_dialog:
                    optimization_dialog.clear()
                    ui.label('AI-Optimized Template').classes('text-h6')
                    
                    with ui.tabs().classes('w-full') as tabs:
                        ui.tab('Subject')
                        ui.tab('Content')
                        ui.tab('Analysis')
                    
                    with ui.tab_panels(tabs).classes('w-full'):
                        with ui.tab_panel('Subject'):
                            ui.label('Original').classes('text-bold')
                            ui.label(current_template.subject)
                            ui.label('Optimized').classes('text-bold text-primary')
                            ui.label(optimized.subject)
                        
                        with ui.tab_panel('Content'):
                            with ui.row().classes('w-full'):
                                with ui.column().classes('w-1/2'):
                                    ui.label('Original').classes('text-bold')
                                    ui.html(current_template.body_content)
                                with ui.column().classes('w-1/2'):
                                    ui.label('Optimized').classes('text-bold')
                                    ui.html(optimized.body)
                        
                        with ui.tab_panel('Analysis'):
                            for key, value in optimized.personalization_strategy.items():
                                ui.label(f'{key}:').classes('text-bold')
                                ui.label(value)
                    
                    with ui.row().classes('w-full justify-end'):
                        ui.button(
                            'Apply Changes',
                            on_click=lambda: apply_optimization(optimized)
                        ).props('primary')
                        ui.button(
                            'Cancel',
                            on_click=optimization_dialog.close
                        ).props('flat')
                
                optimization_dialog.open()
    
    def apply_optimization(optimized):
        subject_input.value = optimized.subject
        body_editor.value = optimized.body
        optimization_dialog.close()
        ui.notify('Optimization applied')
    
    async def refresh_template_list():
        templates = await load_templates()
        template_list.clear()
        
        for template in templates:
            with template_list:
                with ui.card().classes('w-full q-ma-sm'):
                    with ui.row().classes('w-full justify-between items-center'):
                        ui.label(template['name']).classes('text-h6')
                        with ui.row():
                            ui.button(
                                icon='edit',
                                on_click=lambda t=template: load_template(t)
                            ).props('flat')
                            ui.button(
                                icon='delete',
                                on_click=lambda t=template: delete_template(t['id'])
                            ).props('flat negative')
                    
                    ui.label(f"Subject: {template['subject']}")
                    ui.label(f"Created: {template['created_at'].strftime('%Y-%m-%d %H:%M')}")
                    
                    with ui.expansion('Preview').classes('w-full'):
                        ui.html(template['body_content'])
    
    def load_template(template: Dict):
        nonlocal current_template
        current_template = template
        name_input.value = template['name']
        subject_input.value = template['subject']
        body_editor.value = template['body_content']
        ai_checkbox.value = template['is_ai_customizable']
        campaign_select.value = template['campaign_id']
        editor_card.set_visibility(True)
    
    # UI Layout
    ui.label('Email Template Management').classes('text-h4 q-ma-md')
    
    with ui.row().classes('w-full justify-between q-ma-md'):
        ui.button('New Template', on_click=lambda: editor_card.set_visibility(True)).props('primary')
        ui.button('AI Optimize', on_click=optimize_template).props('secondary')
    
    # Template List
    template_list = ui.column().classes('w-full')
    
    # Template Editor
    with ui.card().classes('w-full q-ma-md') as editor_card:
        editor_card.set_visibility(False)
        
        ui.label('Template Editor').classes('text-h6')
        
        name_input = ui.input('Template Name').classes('w-full')
        subject_input = ui.input('Subject Line').classes('w-full')
        
        campaign_select = ui.select(
            'Campaign',
            options=await load_campaigns()  # Implement this helper
        ).classes('w-full')
        
        body_editor = ui.editor(
            placeholder='Email Content (HTML supported)',
            toolbar=['bold', 'italic', 'strike', 'underline', '|',
                    'h1', 'h2', 'h3', '|',
                    'quote', 'unordered', 'ordered', '|',
                    'link', 'clean-block']
        ).classes('w-full')
        
        ai_checkbox = ui.checkbox('Enable AI Customization')
        
        with ui.row().classes('w-full justify-end'):
            ui.button('Save', on_click=save_template).props('primary')
            ui.button('Cancel', on_click=lambda: editor_card.set_visibility(False)).props('flat')
    
    # Dialogs
    optimization_dialog = ui.dialog()
    
    # Load initial data
    await refresh_template_list() 