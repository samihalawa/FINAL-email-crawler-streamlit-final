# EXAMPLE SQL COMMANDS FOR MANUAL UPDATES (DO NOT AUTO-EXECUTE)
DB_UPDATES = {
    "add_email_campaign_columns": """
    -- Add new columns (safe, won't drop existing data)
    ALTER TABLE email_campaigns 
    ADD COLUMN IF NOT EXISTS open_count INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS click_count INTEGER DEFAULT 0;
    """,
    
    "add_campaign_settings": """
    -- Add new columns with defaults
    ALTER TABLE campaigns
    ADD COLUMN IF NOT EXISTS auto_send BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS ai_customization BOOLEAN DEFAULT FALSE;
    """
} 