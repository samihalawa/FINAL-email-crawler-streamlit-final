from nicegui import ui
from typing import Dict, Any
import json
from datetime import datetime

from core.database import get_db, KnowledgeBase
from core.auth import auth_handler
from services.ai import AIService

ai_service = AIService()

@ui.page('/knowledge-base')
@auth_handler.require_auth
async def page():
    # State management
    current_kb: Dict[str, Any] = {}
    
    async def load_knowledge_base():
        async with get_db() as session:
            kb = await session.get(KnowledgeBase, get_active_project_id())
            if kb:
                current_kb.update(kb.content)
                update_kb_display()
    
    async def save_knowledge_base():
        async with get_db() as session:
            kb = await session.get(KnowledgeBase, get_active_project_id())
            if kb:
                kb.content = current_kb
                kb.last_updated = datetime.utcnow()
            else:
                kb = KnowledgeBase(
                    project_id=get_active_project_id(),
                    content=current_kb
                )
                session.add(kb)
            await session.commit()
            ui.notify('Knowledge base saved successfully')
    
    def update_kb_display():
        with kb_display:
            kb_display.clear()
            
            # Business Information
            with ui.expansion('Business Information', value=True).classes('w-full'):
                ui.input(
                    'Company Name',
                    value=current_kb.get('company_name', ''),
                    on_change=lambda e: current_kb.update({'company_name': e.value})
                ).classes('w-full')
                ui.input(
                    'Industry',
                    value=current_kb.get('industry', ''),
                    on_change=lambda e: current_kb.update({'industry': e.value})
                ).classes('w-full')
                ui.textarea(
                    'Company Description',
                    value=current_kb.get('description', ''),
                    on_change=lambda e: current_kb.update({'description': e.value})
                ).classes('w-full')
            
            # Target Audience
            with ui.expansion('Target Audience', value=True).classes('w-full'):
                with ui.row().classes('w-full'):
                    with ui.column().classes('w-1/2'):
                        ui.label('Ideal Customer Profiles')
                        profiles = current_kb.get('ideal_profiles', [])
                        profile_input = ui.input('Add Profile')
                        ui.button('Add', on_click=lambda: add_profile(profile_input.value))
                        for profile in profiles:
                            with ui.row().classes('items-center'):
                                ui.label(profile)
                                ui.button(icon='close', on_click=lambda p=profile: remove_profile(p))
                    
                    with ui.column().classes('w-1/2'):
                        ui.label('Pain Points')
                        pain_points = current_kb.get('pain_points', [])
                        pain_input = ui.input('Add Pain Point')
                        ui.button('Add', on_click=lambda: add_pain_point(pain_input.value))
                        for point in pain_points:
                            with ui.row().classes('items-center'):
                                ui.label(point)
                                ui.button(icon='close', on_click=lambda p=point: remove_pain_point(p))
            
            # Value Proposition
            with ui.expansion('Value Proposition', value=True).classes('w-full'):
                ui.textarea(
                    'Main Value Proposition',
                    value=current_kb.get('value_proposition', ''),
                    on_change=lambda e: current_kb.update({'value_proposition': e.value})
                ).classes('w-full')
                
                benefits = current_kb.get('benefits', [])
                benefit_input = ui.input('Add Benefit')
                ui.button('Add', on_click=lambda: add_benefit(benefit_input.value))
                for benefit in benefits:
                    with ui.row().classes('items-center'):
                        ui.label(benefit)
                        ui.button(icon='close', on_click=lambda b=benefit: remove_benefit(b))
            
            # Communication Style
            with ui.expansion('Communication Style', value=True).classes('w-full'):
                ui.select(
                    'Tone of Voice',
                    options=[
                        'Professional',
                        'Friendly',
                        'Technical',
                        'Casual',
                        'Formal'
                    ],
                    value=current_kb.get('tone', 'Professional'),
                    on_change=lambda e: current_kb.update({'tone': e.value})
                )
                ui.textarea(
                    'Communication Guidelines',
                    value=current_kb.get('guidelines', ''),
                    on_change=lambda e: current_kb.update({'guidelines': e.value})
                ).classes('w-full')
            
            # AI Training Data
            with ui.expansion('AI Training Data', value=True).classes('w-full'):
                ui.textarea(
                    'Custom Instructions',
                    value=current_kb.get('ai_instructions', ''),
                    on_change=lambda e: current_kb.update({'ai_instructions': e.value})
                ).classes('w-full')
                
                examples = current_kb.get('examples', [])
                with ui.dialog() as example_dialog:
                    example_title = ui.input('Title')
                    example_content = ui.textarea('Content')
                    ui.button('Add Example', on_click=lambda: add_example(
                        example_title.value,
                        example_content.value
                    ))
                
                ui.button('Add Example', on_click=example_dialog.open)
                
                for example in examples:
                    with ui.card().classes('w-full q-ma-sm'):
                        ui.label(example['title']).classes('text-bold')
                        ui.label(example['content'])
                        ui.button(
                            icon='close',
                            on_click=lambda e=example: remove_example(e)
                        ).props('flat round')
    
    def add_profile(profile: str):
        if profile:
            profiles = current_kb.get('ideal_profiles', [])
            profiles.append(profile)
            current_kb['ideal_profiles'] = profiles
            update_kb_display()
    
    def remove_profile(profile: str):
        profiles = current_kb.get('ideal_profiles', [])
        profiles.remove(profile)
        current_kb['ideal_profiles'] = profiles
        update_kb_display()
    
    def add_pain_point(point: str):
        if point:
            points = current_kb.get('pain_points', [])
            points.append(point)
            current_kb['pain_points'] = points
            update_kb_display()
    
    def remove_pain_point(point: str):
        points = current_kb.get('pain_points', [])
        points.remove(point)
        current_kb['pain_points'] = points
        update_kb_display()
    
    def add_benefit(benefit: str):
        if benefit:
            benefits = current_kb.get('benefits', [])
            benefits.append(benefit)
            current_kb['benefits'] = benefits
            update_kb_display()
    
    def remove_benefit(benefit: str):
        benefits = current_kb.get('benefits', [])
        benefits.remove(benefit)
        current_kb['benefits'] = benefits
        update_kb_display()
    
    def add_example(title: str, content: str):
        if title and content:
            examples = current_kb.get('examples', [])
            examples.append({
                'title': title,
                'content': content,
                'created_at': datetime.utcnow().isoformat()
            })
            current_kb['examples'] = examples
            update_kb_display()
    
    def remove_example(example: Dict):
        examples = current_kb.get('examples', [])
        examples.remove(example)
        current_kb['examples'] = examples
        update_kb_display()
    
    async def analyze_and_suggest():
        """Use AI to analyze KB and suggest improvements"""
        with ui.loading(message='Analyzing knowledge base...'):
            suggestions = await ai_service.analyze_knowledge_base(current_kb)
            
            with ui.dialog() as suggestion_dialog:
                ui.label('AI Suggestions').classes('text-h6')
                
                for category, items in suggestions.items():
                    with ui.expansion(category, value=True):
                        for item in items:
                            with ui.card().classes('w-full q-ma-sm'):
                                ui.label(item['suggestion']).classes('text-bold')
                                ui.label(item['rationale'])
                                if item.get('example'):
                                    ui.label(f"Example: {item['example']}")
                                if item.get('action'):
                                    ui.button(
                                        'Apply',
                                        on_click=lambda a=item['action']: apply_suggestion(a)
                                    ).props('outline')
                
            suggestion_dialog.open()
    
    async def apply_suggestion(action: Dict):
        """Apply AI suggestion to knowledge base"""
        current_kb.update(action)
        update_kb_display()
        ui.notify('Suggestion applied successfully')
    
    async def export_kb():
        """Export knowledge base as JSON"""
        ui.download(
            json.dumps(current_kb, indent=2),
            'knowledge_base.json'
        )
    
    # UI Layout
    ui.label('Knowledge Base Management').classes('text-h4 q-ma-md')
    
    with ui.row().classes('w-full justify-between q-ma-md'):
        ui.button('Save Changes', on_click=save_knowledge_base).props('primary')
        ui.button('AI Analysis', on_click=analyze_and_suggest).props('secondary')
        ui.button('Export', on_click=export_kb).props('outline')
    
    kb_display = ui.column().classes('w-full')
    
    # Load initial data
    await load_knowledge_base() 