from nicegui import ui
from typing import Dict, Any, List
from datetime import datetime
import plotly.express as px
import pandas as pd

from core.database import get_db, SearchTerm, SearchTermGroup
from core.auth import auth_handler
from services.ai import AIService
from services.search import SearchService

ai_service = AIService()
search_service = SearchService()

@ui.page('/search-terms')
@auth_handler.require_auth
async def page():
    # State management
    selected_terms: List[int] = []
    current_group: Dict[str, Any] = {}
    
    async def load_terms():
        """Load search terms with performance metrics"""
        async with get_db() as session:
            terms = await session.execute("""
                SELECT 
                    st.*,
                    stg.name as group_name,
                    COUNT(DISTINCT l.id) as total_leads,
                    AVG(CASE WHEN l.id IS NOT NULL THEN 1 ELSE 0 END) as success_rate
                FROM search_terms st
                LEFT JOIN search_term_groups stg ON st.group_id = stg.id
                LEFT JOIN leads l ON l.source_term_id = st.id
                GROUP BY st.id
                ORDER BY st.created_at DESC
            """)
            return [dict(t) for t in terms]
    
    async def load_groups():
        """Load search term groups"""
        async with get_db() as session:
            groups = await session.execute("""
                SELECT 
                    stg.*,
                    COUNT(DISTINCT st.id) as term_count,
                    COUNT(DISTINCT l.id) as total_leads
                FROM search_term_groups stg
                LEFT JOIN search_terms st ON st.group_id = stg.id
                LEFT JOIN leads l ON l.source_term_id = st.id
                GROUP BY stg.id
                ORDER BY stg.name
            """)
            return [dict(g) for g in groups]
    
    async def optimize_terms():
        """Optimize selected search terms with AI"""
        if not selected_terms:
            ui.notify('Please select terms to optimize', type='warning')
            return
        
        async with get_db() as session:
            terms = []
            for term_id in selected_terms:
                term = await session.get(SearchTerm, term_id)
                if term:
                    terms.append(term)
            
            # Get knowledge base for context
            kb = await session.get_knowledge_base()
            if not kb:
                ui.notify('Knowledge base not found', type='negative')
                return
            
            with ui.loading('Optimizing search terms...'):
                strategy = await ai_service.generate_search_strategy(
                    kb,
                    [t.term for t in terms]
                )
                
                with optimization_dialog:
                    optimization_dialog.clear()
                    ui.label('Search Term Optimization').classes('text-h6')
                    
                    with ui.tabs().classes('w-full') as tabs:
                        ui.tab('Optimized Terms')
                        ui.tab('Strategy')
                        ui.tab('Analysis')
                    
                    with ui.tab_panels(tabs).classes('w-full'):
                        # Optimized Terms Panel
                        with ui.tab_panel('Optimized Terms'):
                            ui.label('Original vs Optimized Terms').classes('text-bold')
                            
                            for i, term in enumerate(terms):
                                with ui.card().classes('w-full q-ma-sm'):
                                    ui.label('Original').classes('text-bold')
                                    ui.label(term.term)
                                    ui.label('Optimized').classes('text-bold text-primary')
                                    ui.label(strategy.search_terms[i])
                                    
                                    ui.checkbox(
                                        'Apply Optimization',
                                        value=True,
                                        on_change=lambda e, t=term, opt=strategy.search_terms[i]: toggle_term_optimization(t.id, opt, e.value)
                                    )
                        
                        # Strategy Panel
                        with ui.tab_panel('Strategy'):
                            ui.label('Optimization Strategy').classes('text-bold')
                            ui.label(strategy.rationale)
                            
                            ui.label('Target Audience').classes('text-bold q-mt-md')
                            for key, value in strategy.target_audience.items():
                                ui.label(f"{key}: {value}")
                        
                        # Analysis Panel
                        with ui.tab_panel('Analysis'):
                            performance_data = []
                            for term in terms:
                                performance = await analyze_term_performance(term.id)
                                performance_data.append({
                                    'term': term.term,
                                    'leads': performance['total_leads'],
                                    'success_rate': performance['success_rate'],
                                    'quality_score': performance['quality_score']
                                })
                            
                            df = pd.DataFrame(performance_data)
                            fig = px.bar(
                                df,
                                x='term',
                                y=['leads', 'success_rate', 'quality_score'],
                                title='Term Performance Metrics',
                                barmode='group'
                            )
                            ui.plotly(fig).classes('w-full h-64')
                    
                    with ui.row().classes('w-full justify-end'):
                        ui.button(
                            'Apply Selected Optimizations',
                            on_click=apply_optimizations
                        ).props('primary')
                        ui.button(
                            'Cancel',
                            on_click=optimization_dialog.close
                        ).props('flat')
                
                optimization_dialog.open()
    
    async def analyze_term_performance(term_id: int):
        """Analyze performance of a search term"""
        async with get_db() as session:
            results = await session.execute("""
                SELECT 
                    COUNT(DISTINCT l.id) as total_leads,
                    AVG(CASE WHEN l.id IS NOT NULL THEN 1 ELSE 0 END) as success_rate,
                    AVG(CASE 
                        WHEN l.company IS NOT NULL AND l.position IS NOT NULL THEN 1
                        WHEN l.company IS NOT NULL OR l.position IS NOT NULL THEN 0.5
                        ELSE 0 
                    END) as quality_score
                FROM search_terms st
                LEFT JOIN leads l ON l.source_term_id = st.id
                WHERE st.id = :term_id
            """, {'term_id': term_id})
            return dict(results.first())
    
    def toggle_term_optimization(term_id: int, optimized_term: str, apply: bool):
        """Toggle term optimization selection"""
        if apply:
            term_optimizations[term_id] = optimized_term
        elif term_id in term_optimizations:
            del term_optimizations[term_id]
    
    async def apply_optimizations():
        """Apply selected term optimizations"""
        async with get_db() as session:
            for term_id, optimized_term in term_optimizations.items():
                term = await session.get(SearchTerm, term_id)
                if term:
                    term.term = optimized_term
            await session.commit()
            ui.notify('Terms optimized successfully')
            optimization_dialog.close()
            await refresh_terms()
    
    async def create_group():
        """Create new search term group"""
        async with get_db() as session:
            group = SearchTermGroup(
                name=group_name_input.value,
                description=group_description_input.value
            )
            session.add(group)
            await session.commit()
            ui.notify('Group created successfully')
            group_dialog.close()
            await refresh_groups()
    
    async def assign_to_group():
        """Assign selected terms to group"""
        if not selected_terms:
            ui.notify('Please select terms to assign', type='warning')
            return
        
        async with get_db() as session:
            for term_id in selected_terms:
                term = await session.get(SearchTerm, term_id)
                if term:
                    term.group_id = group_select.value
            await session.commit()
            ui.notify('Terms assigned successfully')
            await refresh_terms()
    
    def update_term_table(terms: List[Dict]):
        """Update the search terms table"""
        with term_table:
            term_table.clear()
            
            with ui.table().props('rows-per-page-options=[10,20,50]').classes('w-full'):
                ui.table_column('Select', 'select')
                ui.table_column('Term', 'term')
                ui.table_column('Group', 'group_name')
                ui.table_column('Total Leads', 'total_leads')
                ui.table_column('Success Rate', 'success_rate')
                ui.table_column('Actions', 'actions')
                
                rows = []
                for term in terms:
                    row = {
                        'select': ui.checkbox(
                            value=term['id'] in selected_terms,
                            on_change=lambda e, t=term: toggle_term_selection(t['id'], e.value)
                        ),
                        'term': term['term'],
                        'group_name': term['group_name'] or 'Ungrouped',
                        'total_leads': term['total_leads'],
                        'success_rate': f"{term['success_rate']*100:.1f}%",
                        'actions': ui.button(
                            'View Performance',
                            on_click=lambda t=term: show_term_performance(t['id'])
                        ).props('outline')
                    }
                    rows.append(row)
                
                ui.table_rows(rows)
    
    def toggle_term_selection(term_id: int, selected: bool):
        """Toggle term selection"""
        if selected and term_id not in selected_terms:
            selected_terms.append(term_id)
        elif not selected and term_id in selected_terms:
            selected_terms.remove(term_id)
    
    async def show_term_performance(term_id: int):
        """Show detailed term performance"""
        async with get_db() as session:
            term = await session.get(SearchTerm, term_id)
            if not term:
                ui.notify('Term not found', type='negative')
                return
            
            # Get performance data
            performance = await analyze_term_performance(term_id)
            
            # Get leads over time
            leads_over_time = await session.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM leads
                WHERE source_term_id = :term_id
                GROUP BY DATE(created_at)
                ORDER BY date
            """, {'term_id': term_id})
            
            with performance_dialog:
                performance_dialog.clear()
                ui.label(f"Term Performance: {term.term}").classes('text-h6')
                
                with ui.tabs().classes('w-full') as tabs:
                    ui.tab('Overview')
                    ui.tab('Lead Quality')
                    ui.tab('Trends')
                
                with ui.tab_panels(tabs).classes('w-full'):
                    # Overview Panel
                    with ui.tab_panel('Overview'):
                        with ui.row().classes('w-full justify-between'):
                            with ui.card().classes('w-1/3'):
                                ui.label('Total Leads').classes('text-h6')
                                ui.label(str(performance['total_leads']))
                            
                            with ui.card().classes('w-1/3'):
                                ui.label('Success Rate').classes('text-h6')
                                ui.label(f"{performance['success_rate']*100:.1f}%")
                            
                            with ui.card().classes('w-1/3'):
                                ui.label('Quality Score').classes('text-h6')
                                ui.label(f"{performance['quality_score']*100:.1f}%")
                    
                    # Lead Quality Panel
                    with ui.tab_panel('Lead Quality'):
                        quality_data = await session.execute("""
                            SELECT 
                                CASE 
                                    WHEN company IS NOT NULL AND position IS NOT NULL THEN 'High'
                                    WHEN company IS NOT NULL OR position IS NOT NULL THEN 'Medium'
                                    ELSE 'Low'
                                END as quality,
                                COUNT(*) as count
                            FROM leads
                            WHERE source_term_id = :term_id
                            GROUP BY quality
                        """, {'term_id': term_id})
                        
                        df_quality = pd.DataFrame([dict(r) for r in quality_data])
                        if not df_quality.empty:
                            fig = px.pie(
                                df_quality,
                                values='count',
                                names='quality',
                                title='Lead Quality Distribution'
                            )
                            ui.plotly(fig).classes('w-full h-64')
                    
                    # Trends Panel
                    with ui.tab_panel('Trends'):
                        df_trends = pd.DataFrame([dict(r) for r in leads_over_time])
                        if not df_trends.empty:
                            fig = px.line(
                                df_trends,
                                x='date',
                                y='count',
                                title='Leads Generated Over Time'
                            )
                            ui.plotly(fig).classes('w-full h-64')
                
                performance_dialog.open()
    
    async def refresh_terms():
        """Refresh terms list"""
        terms = await load_terms()
        update_term_table(terms)
    
    async def refresh_groups():
        """Refresh groups list"""
        groups = await load_groups()
        group_select.options = [{'label': g['name'], 'value': g['id']} for g in groups]
    
    # UI Layout
    ui.label('Search Terms Management').classes('text-h4 q-ma-md')
    
    # Action Buttons
    with ui.row().classes('w-full justify-between q-ma-md'):
        with ui.row():
            ui.button('Optimize Selected', on_click=optimize_terms).props('secondary')
            ui.button('Create Group', on_click=lambda: group_dialog.open()).props('primary')
            ui.button('Delete Selected', on_click=delete_selected).props('negative outline')
        
        group_select = ui.select(
            'Assign to Group',
            options=await load_groups(),
            on_change=assign_to_group
        ).classes('w-1/3')
    
    # Terms Table
    term_table = ui.column().classes('w-full')
    
    # Dialogs
    optimization_dialog = ui.dialog()
    performance_dialog = ui.dialog()
    
    with ui.dialog() as group_dialog:
        ui.label('Create Search Term Group').classes('text-h6')
        group_name_input = ui.input('Group Name').classes('w-full')
        group_description_input = ui.textarea('Description').classes('w-full')
        
        with ui.row().classes('w-full justify-end'):
            ui.button('Create', on_click=create_group).props('primary')
            ui.button('Cancel', on_click=group_dialog.close).props('flat')
    
    # State for optimizations
    term_optimizations = {}
    
    # Load initial data
    await refresh_terms() 