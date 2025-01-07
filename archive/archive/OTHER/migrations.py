from alembic import op
import sqlalchemy as sa
from datetime import datetime

def upgrade():
    connection = op.get_bind()
    
    # Create base tables
    op.create_table(
        'projects',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('project_name', sa.Text(), default="Default Project"),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    op.create_table(
        'campaigns',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('campaign_name', sa.Text(), default="Default Campaign"),
        sa.Column('campaign_type', sa.Text(), default="Email"),
        sa.Column('project_id', sa.BigInteger(), sa.ForeignKey('projects.id'), default=1),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('auto_send', sa.Boolean(), default=False),
        sa.Column('loop_automation', sa.Boolean(), default=False),
        sa.Column('ai_customization', sa.Boolean(), default=False),
        sa.Column('max_emails_per_group', sa.BigInteger(), default=40),
        sa.Column('loop_interval', sa.BigInteger(), default=60)
    )

    op.create_table(
        'knowledge_base',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('project_id', sa.BigInteger(), sa.ForeignKey('projects.id'), nullable=False),
        sa.Column('kb_name', sa.Text()),
        sa.Column('kb_bio', sa.Text()),
        sa.Column('kb_values', sa.Text()),
        sa.Column('contact_name', sa.Text()),
        sa.Column('contact_role', sa.Text()),
        sa.Column('contact_email', sa.Text()),
        sa.Column('company_description', sa.Text()),
        sa.Column('company_mission', sa.Text()),
        sa.Column('company_target_market', sa.Text()),
        sa.Column('company_other', sa.Text()),
        sa.Column('product_name', sa.Text()),
        sa.Column('product_description', sa.Text()),
        sa.Column('product_target_customer', sa.Text()),
        sa.Column('product_other', sa.Text()),
        sa.Column('other_context', sa.Text()),
        sa.Column('example_email', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now())
    )

    op.create_table(
        'leads',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('email', sa.Text(), unique=True),
        sa.Column('phone', sa.Text()),
        sa.Column('first_name', sa.Text()),
        sa.Column('last_name', sa.Text()),
        sa.Column('company', sa.Text()),
        sa.Column('job_title', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    op.create_table(
        'search_term_groups',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('name', sa.Text()),
        sa.Column('email_template', sa.Text()),
        sa.Column('description', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    op.create_table(
        'search_terms',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('group_id', sa.BigInteger(), sa.ForeignKey('search_term_groups.id')),
        sa.Column('campaign_id', sa.BigInteger(), sa.ForeignKey('campaigns.id')),
        sa.Column('term', sa.Text()),
        sa.Column('category', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('language', sa.Text(), default='ES')
    )

    op.create_table(
        'campaign_leads',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('campaign_id', sa.BigInteger(), sa.ForeignKey('campaigns.id')),
        sa.Column('lead_id', sa.BigInteger(), sa.ForeignKey('leads.id')),
        sa.Column('status', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    op.create_table(
        'email_templates',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('campaign_id', sa.BigInteger(), sa.ForeignKey('campaigns.id')),
        sa.Column('template_name', sa.Text()),
        sa.Column('subject', sa.Text()),
        sa.Column('body_content', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('is_ai_customizable', sa.Boolean(), default=False),
        sa.Column('language', sa.Text(), default='ES')
    )

    op.create_table(
        'email_campaigns',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('campaign_id', sa.BigInteger(), sa.ForeignKey('campaigns.id')),
        sa.Column('lead_id', sa.BigInteger(), sa.ForeignKey('leads.id')),
        sa.Column('template_id', sa.BigInteger(), sa.ForeignKey('email_templates.id')),
        sa.Column('customized_subject', sa.Text()),
        sa.Column('customized_content', sa.Text()),
        sa.Column('original_subject', sa.Text()),
        sa.Column('original_content', sa.Text()),
        sa.Column('status', sa.Text()),
        sa.Column('engagement_data', sa.JSON()),
        sa.Column('message_id', sa.Text()),
        sa.Column('tracking_id', sa.Text(), unique=True),
        sa.Column('sent_at', sa.DateTime(timezone=True)),
        sa.Column('ai_customized', sa.Boolean(), default=False),
        sa.Column('opened_at', sa.DateTime(timezone=True)),
        sa.Column('clicked_at', sa.DateTime(timezone=True)),
        sa.Column('open_count', sa.BigInteger(), default=0),
        sa.Column('click_count', sa.BigInteger(), default=0)
    )

    op.create_table(
        'lead_sources',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('lead_id', sa.BigInteger(), sa.ForeignKey('leads.id')),
        sa.Column('search_term_id', sa.BigInteger(), sa.ForeignKey('search_terms.id')),
        sa.Column('url', sa.Text()),
        sa.Column('domain', sa.Text()),
        sa.Column('page_title', sa.Text()),
        sa.Column('meta_description', sa.Text()),
        sa.Column('scrape_duration', sa.Text()),
        sa.Column('meta_tags', sa.Text()),
        sa.Column('phone_numbers', sa.Text()),
        sa.Column('content', sa.Text()),
        sa.Column('tags', sa.Text()),
        sa.Column('http_status', sa.BigInteger()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    op.create_table(
        'settings',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('setting_type', sa.Text(), nullable=False),
        sa.Column('value', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now())
    )

    op.create_table(
        'email_settings',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('email', sa.Text(), nullable=False),
        sa.Column('provider', sa.Text(), nullable=False),
        sa.Column('smtp_server', sa.Text()),
        sa.Column('smtp_port', sa.BigInteger()),
        sa.Column('smtp_username', sa.Text()),
        sa.Column('smtp_password', sa.Text()),
        sa.Column('aws_access_key_id', sa.Text()),
        sa.Column('aws_secret_access_key', sa.Text()),
        sa.Column('aws_region', sa.Text()),
        sa.Column('daily_limit', sa.BigInteger(), default=200)
    )

    op.create_table(
        'email_quotas',
        sa.Column('id', sa.BigInteger(), primary_key=True),
        sa.Column('email_settings_id', sa.BigInteger(), sa.ForeignKey('email_settings.id', ondelete='CASCADE')),
        sa.Column('emails_sent_today', sa.BigInteger(), default=0),
        sa.Column('last_reset', sa.DateTime(timezone=True)),
        sa.Column('error_count', sa.BigInteger(), default=0),
        sa.Column('last_error', sa.Text()),
        sa.Column('last_error_time', sa.DateTime(timezone=True)),
        sa.Column('lock_version', sa.BigInteger(), default=0)
    )

    # Create indexes
    op.create_index('idx_email_quota_settings', 'email_quotas', ['email_settings_id'])
    op.create_index('idx_email_quota_reset', 'email_quotas', ['last_reset'])
    op.create_index('idx_leads_email', 'leads', ['email'])
    op.create_index('idx_lead_sources_domain', 'lead_sources', ['domain'])
    op.create_index('idx_email_campaigns_tracking', 'email_campaigns', ['tracking_id'])

def downgrade():
    # Drop tables in reverse order of creation
    tables = [
        'email_quotas', 'email_settings', 'settings', 'lead_sources',
        'email_campaigns', 'email_templates', 'campaign_leads', 'search_terms',
        'search_term_groups', 'leads', 'knowledge_base', 'campaigns', 'projects'
    ]
    
    for table in tables:
        op.drop_table(table)