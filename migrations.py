from alembic import op
import sqlalchemy as sa
from datetime import datetime

def upgrade():
    # First check if tables already exist to prevent errors
    connection = op.get_bind()
    
    # Check if email_quotas table already exists
    has_quota_table = connection.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'email_quotas'
        );
    """).scalar()
    
    if not has_quota_table:
        # Create new table only if it doesn't exist
        op.create_table(
            'email_quotas',
            sa.Column('id', sa.BigInteger(), nullable=False),
            sa.Column('email_settings_id', sa.BigInteger(), nullable=False),
            sa.Column('emails_sent_today', sa.BigInteger(), default=0),
            sa.Column('last_reset', sa.DateTime(timezone=True)),
            sa.Column('error_count', sa.BigInteger(), default=0),
            sa.Column('last_error', sa.Text()),
            sa.Column('last_error_time', sa.DateTime(timezone=True)),
            sa.ForeignKeyConstraint(['email_settings_id'], ['email_settings.id']),
            sa.PrimaryKeyConstraint('id')
        )

    # Check if daily_limit column already exists
    has_daily_limit = connection.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'email_settings' AND column_name = 'daily_limit'
        );
    """).scalar()
    
    if not has_daily_limit:
        # Add daily_limit column only if it doesn't exist
        op.add_column('email_settings',
            sa.Column('daily_limit', sa.BigInteger(), server_default='200')
        )

    # Initialize quotas only for settings that don't have them
    now = datetime.utcnow()
    email_settings = connection.execute(
        """
        SELECT es.id 
        FROM email_settings es
        LEFT JOIN email_quotas eq ON es.id = eq.email_settings_id
        WHERE eq.id IS NULL
        """
    ).fetchall()
    
    for settings_id in email_settings:
        connection.execute(
            f"""
            INSERT INTO email_quotas 
            (email_settings_id, emails_sent_today, last_reset, error_count) 
            VALUES 
            ({settings_id[0]}, 0, '{now}', 0)
            """
        )

def downgrade():
    # We'll make the downgrade safer by checking first
    connection = op.get_bind()
    
    # Only drop if tables/columns exist
    has_quota_table = connection.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'email_quotas'
        );
    """).scalar()
    
    has_daily_limit = connection.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'email_settings' AND column_name = 'daily_limit'
        );
    """).scalar()
    
    if has_quota_table:
        op.drop_table('email_quotas')
    
    if has_daily_limit:
        op.drop_column('email_settings', 'daily_limit') 